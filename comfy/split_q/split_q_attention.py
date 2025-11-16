"""
Split-Q attention implementation
Moved from comfy/ldm/modules/attention.py
"""

import math
import torch
from torch import einsum
import logging

from comfy.ldm.modules.attention import wrap_attn, get_attn_precision
import comfy.model_management as model_management

# Deduplication for logging
_last_log_messages = []

def _log_once(message):
    global _last_log_messages
    if message not in _last_log_messages:
        logging.info(message)
        _last_log_messages.append(message)
        if len(_last_log_messages) > 10:
            _last_log_messages.pop(0)


@wrap_attn
def attention_split(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    attn_precision = get_attn_precision(attn_precision, q.dtype)

    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    scale = dim_head ** -0.5

    if skip_reshape:
         q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    mem_free_total = model_management.get_free_memory(q.device)

    if attn_precision == torch.float32:
        element_size = 4
        upcast = True
    else:
        element_size = q.element_size()
        upcast = False

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * element_size
    modifier = 3
    mem_required = tensor_size * modifier
    steps = 2
    _log_once(f"\n⚡ [split-q]  = {steps}")

    if mask is not None:
        if len(mask.shape) == 2:
            bs = 1
        else:
            bs = mask.shape[0]
        mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])

    # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
    first_op_done = False
    cleared_cache = False
    while True:
        try:
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            _log_once(f"\n⚡ [split-q] slice_size = {slice_size}")
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                if upcast:
                    with torch.autocast(enabled=False, device_type = 'cuda'):
                        s1 = einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * scale
                else:
                    s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * scale

                if mask is not None:
                    if len(mask.shape) == 2:
                        s1 += mask[i:end]
                    else:
                        if mask.shape[1] == 1:
                            s1 += mask
                        else:
                            s1 += mask[:, i:end]

                s2 = s1.softmax(dim=-1).to(v.dtype)
                del s1
                first_op_done = True

                r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                del s2
            break
        except model_management.OOM_EXCEPTION as e:
            if first_op_done == False:
                model_management.soft_empty_cache(True)
                if cleared_cache == False:
                    cleared_cache = True
                    logging.warning("out of memory error, emptying cache and trying again")
                    continue
                steps *= 2
                if steps > 64:
                    raise e
                logging.warning("out of memory error, increasing steps and trying again {}".format(steps))
            else:
                raise e

    del q, k, v

    if skip_output_reshape:
        r1 = (
            r1.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
        )
    else:
        r1 = (
            r1.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    return r1