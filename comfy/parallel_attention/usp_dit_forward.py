"""Unified Sequence Parallel DiT orchestrator."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from comfy.ldm.flux.layers import timestep_embedding

logger = logging.getLogger(__name__)

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from .usp_double_forward import usp_double_forward
from .usp_single_forward import usp_single_forward

LOG_PREFIX = "⚡ [Parallel-Attention][USP][Orchestrator]"
LOGGER = logging.getLogger(__name__)


def pad_if_odd(t: Tensor, dim: int = 1) -> Tensor:
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


def usp_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Optional[Tensor] = None,
    control=None,
    transformer_options: Optional[dict] = None,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    # NOP: STEP5.0 debug log removed for performance
    
    transformer_options = transformer_options or {}
    patches_replace = transformer_options.get("patches_replace", {})

    img = pad_if_odd(img, dim=1)
    if img_ids is not None:
        img_ids = pad_if_odd(img_ids, dim=1)
    txt = pad_if_odd(txt, dim=1)
    if txt_ids is not None:
        txt_ids = pad_if_odd(txt_ids, dim=1)
    
    # NOP: STEP5.1 debug log removed for performance

    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
    
    # CHECKPOINT: STEP3 - initial noisy latent (logging removed)

    img = self.img_in(img)
    
    # CHECKPOINT: STEP4 - after img_in projection (logging removed)
    
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    
    # CHECKPOINT: STEP5 - vec after timestep+guidance (logging removed)

    vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])
    txt = self.txt_in(txt)
    
    # CHECKPOINT: STEP2 - encoded prompt after txt_in (logging removed)

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe_combine = self.pe_embedder(ids)
        pe_image = self.pe_embedder(img_ids)
        
        # CHECKPOINT: STEP6 - pe_image before chunking (logging removed)
        
        # Chunk PE along dim=2 (features) for Ulysses parallelism
        pe_combine = torch.chunk(pe_combine, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
        pe_image = torch.chunk(pe_image, get_sequence_parallel_world_size(), dim=2)[get_sequence_parallel_rank()]
    else:
        pe_combine = None
        pe_image = None

    # CHECKPOINT: STEP7 - img BEFORE chunking (logging removed)
    
    # Chunk sequences along dim=1 for each rank
    img = torch.chunk(img, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    txt = torch.chunk(txt, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    
    # CHECKPOINT: STEP8 - img after ulysses chunk (logging removed)

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        block_key = ("double_block", i)
        if block_key in blocks_replace:
            def block_wrap(args):
                out_img, out_txt = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                )
                return {"img": out_img, "txt": out_txt}

            patched = blocks_replace[block_key](
                {
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe_image,
                    "attn_mask": attn_mask,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            img = patched["img"]
            txt = patched["txt"]
        else:
            img, txt = block(
                img=img,
                txt=txt,
                vec=vec,
                pe=pe_image,
                attn_mask=attn_mask,
            )

        if control is not None:
            control_i = control.get("input")
            if control_i and i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    # CHECKPOINT: STEP17 - Before all-gather (end of double blocks)

    # Gather full sequences from all ranks
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    txt = get_sp_group().all_gather(txt.contiguous(), dim=1)
    
    # CHECKPOINT: STEP18 - After all-gather (txt+img sequence reconstruction)

    # Concatenate and rechunk for single blocks
    txt_len = txt.shape[1]
    combined = torch.cat((txt, img), dim=1)
    combined = torch.chunk(combined, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    
    # STEP 18.3 VERIFICATION: pe_combine before single blocks
    if get_sequence_parallel_rank() == 0:
        import hashlib
        pe_combine_bytes = pe_combine.detach().cpu().to(torch.float32).numpy().tobytes()
        pe_combine_hash = hashlib.sha256(pe_combine_bytes).hexdigest()[:16]
        logger.info("⚡ [STEP18.3-VERIFY] pe_combine BEFORE single blocks - hash: %s, shape: %s, pe_combine[0,0,100,30]=%s",
                    pe_combine_hash, pe_combine.shape, pe_combine[0,0,100,30].tolist())

    for i, block in enumerate(self.single_blocks):
        block_key = ("single_block", i)
        if block_key in blocks_replace:
            def block_wrap(args):
                out = block(
                    args["img"],
                    vec=args["vec"],
                    pe=args["pe"],
                    attn_mask=args.get("attn_mask"),
                )
                return {"img": out}

            patched = blocks_replace[block_key](
                {
                    "img": combined,
                    "vec": vec,
                    "pe": pe_combine,
                    "attn_mask": attn_mask,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )
            combined = patched["img"]
        else:
            combined = block(
                combined,
                vec=vec,
                pe=pe_combine,
                attn_mask=attn_mask,
            )

        # CHECKPOINT: STEP18.5 - After first single block (critical divergence point)

        if control is not None:
            control_o = control.get("output")
            if control_o and i < len(control_o):
                add = control_o[i]
                if add is not None:
                    combined[:, txt_len:] += add

    # CHECKPOINT: STEP19 - After all single blocks, before final gather
    
    # Gather full combined sequence
    combined = get_sp_group().all_gather(combined, dim=1)
    
    # CHECKPOINT: STEP20 - After final all-gather (combined sequence reconstruction)
    
    result = combined[:, txt_len:]

    return self.final_layer(result, vec)
