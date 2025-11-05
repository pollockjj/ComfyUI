"""Ring-Attention patches using ComfyUI's patches/patches_replace mechanism.

This module implements the "Chained Patch" strategy to inject Ring-Attention
sequence parallelism at three critical points in Flux's forward_orig method.

Architecture:
    Point 1 (Split): patches["post_input"] - Split img/txt after projections
    Point 2 (Gather/Resplit): patches_replace chain to bridge the torch.cat gap
    Point 3 (Final Gather): patches_replace["single_block", M-1] - Final output

Reference: Research report "ComfyUI Patches for Ring-Attention.md"
"""

import torch
import torch.distributed as dist
import logging
from typing import Dict, Any, Optional

LOG_PREFIX = "⚡ [RingPatches]"


def all_gather_sequence(
    tensor: torch.Tensor, 
    group: dist.ProcessGroup, 
    dim: int = 1
) -> torch.Tensor:
    """All-gather tensor along sequence dimension.
    
    Args:
        tensor: Local tensor chunk [batch, local_seq, channels]
        group: Distributed process group
        dim: Dimension to concatenate along (default: 1 for sequence)
    
    Returns:
        Full tensor [batch, full_seq, channels]
    """
    world_size = dist.get_world_size(group)
    
    # Gather from all ranks
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    
    # Concatenate along sequence dimension
    return torch.cat(gathered, dim=dim)


class RingAttentionPatches:
    """Ring-Attention patch functions for Flux forward_orig.
    
    This class provides three patch functions that implement the "Chained Patch"
    strategy to inject Ring-Attention split/gather logic without modifying
    ComfyUI core files.
    """
    
    def __init__(self, depth_double: int = 19, depth_single: int = 38):
        """Initialize with Flux model block counts.
        
        Args:
            depth_double: Number of double_blocks (default: 19 for Flux-Dev)
            depth_single: Number of single_blocks (default: 38 for Flux-Dev)
        """
        self.depth_double = depth_double
        self.depth_single = depth_single
        
        # Block indices for patching
        self.last_double_idx = depth_double - 1
        self.first_single_idx = 0
        self.last_single_idx = depth_single - 1
    
    # ========================================================================
    # POINT 1: Initial Split (patches_replace on double_block 0)
    # ========================================================================
    
    def patch_first_double_block(
        self, 
        args: Dict[str, Any], 
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Split img/txt before first double_block (Point 1).
        
        API: patches_replace (Override)
        Hook: patches_replace["dit"][("double_block", 0)]
        
        This runs INSTEAD of double_block[0], allowing us to split the input
        before processing, then call the original block with split tensors.
        
        Args:
            args: {
                "img": [batch, full_seq, hidden],
                "txt": [batch, txt_len, hidden],
                "vec": timestep embeddings,
                "pe": positional embeddings,
                "transformer_options": state bag (HAS ring_context!)
            }
            objects: {"original_block": callable}
        
        Returns:
            {"img": split_output, "txt": split_output}
        """
        transformer_options = args.get("transformer_options", {})
        ring_context = transformer_options.get("ring_context")
        
        if not ring_context:
            logging.error(f"{LOG_PREFIX} patch_first_double_block: No ring_context!")
            # Fallback: call original block
            return objects["original_block"](args)
        
        rank = ring_context.get("rank")
        world_size = ring_context.get("world_size")
        
        if rank is None:
            logging.error(f"{LOG_PREFIX} patch_first_double_block: rank is None!")
            return objects["original_block"](args)
        
        # Split img, txt AND positional embeddings (pe)
        img = args["img"]
        txt = args["txt"]
        pe = args.get("pe")  # Positional embeddings (combined txt_ids + img_ids)
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] ✅ FIRST DOUBLE_BLOCK: Splitting input: "
            f"img {img.shape}, txt {txt.shape}, pe {pe.shape if pe is not None else 'None'}"
        )
        
        img_local = torch.chunk(img, world_size, dim=1)[rank].contiguous()
        txt_local = torch.chunk(txt, world_size, dim=1)[rank].contiguous()
        
        # Split pe if it exists (it's combined txt+img positional embeddings)
        # pe shape: [batch, 1, seq_len, 64, 2, 2] - split on dim 2
        if pe is not None:
            pe_local = torch.chunk(pe, world_size, dim=2)[rank].contiguous()
            args["pe"] = pe_local
            logging.info(f"{LOG_PREFIX}[Rank {rank}] Split pe: {pe.shape} → {pe_local.shape}")
        
        # Update args with split tensors
        args["img"] = img_local
        args["txt"] = txt_local
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] Split complete: "
            f"img={img_local.shape}, txt={txt_local.shape}, "
            f"pe={args['pe'].shape if pe is not None else 'None'}"
        )
        
        # Store split pe in transformer_options for other blocks
        # CRITICAL: Modify args["transformer_options"] directly so ALL blocks see it
        if "transformer_options" in args:
            args["transformer_options"]["ring_split_pe"] = args["pe"]
            logging.info(
                f"{LOG_PREFIX}[Rank {rank}] ✅ Stored split pe in transformer_options for blocks 1-18"
            )
        
        # Call original double_block[0] with split tensors
        return objects["original_block"](args)
    
    def patch_middle_double_block(
        self,
        args: Dict[str, Any],
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use split pe from transformer_options for middle double_blocks (1-17).
        
        API: patches_replace (Override)
        Hook: patches_replace["dit"][("double_block", 1-17)]
        
        Middle blocks (1-17) execute with the OUTER scope's pe, which is still
        full-size (4352). We need to replace it with the split pe (2176) that
        was stored by patch_first_double_block.
        
        Args:
            args: {"img": tensor, "txt": tensor, "vec": tensor, "pe": FULL tensor}
            objects: {"original_block": callable}
        
        Returns:
            {"img": output, "txt": output} from original block
        """
        transformer_options = args.get("transformer_options", {})
        ring_split_pe = transformer_options.get("ring_split_pe")
        ring_context = transformer_options.get("ring_context", {})
        rank = ring_context.get("rank", "?")
        
        if ring_split_pe is not None:
            # Replace full pe with split pe
            logging.debug(
                f"{LOG_PREFIX}[Rank {rank}] Middle double_block: replacing pe {args['pe'].shape} with split {ring_split_pe.shape}"
            )
            args["pe"] = ring_split_pe
        else:
            logging.warning(
                f"{LOG_PREFIX}[Rank {rank}] Middle double_block: ring_split_pe NOT FOUND in transformer_options!"
            )
        
        # Execute original block with split pe
        return objects["original_block"](args)
    
    def patch_post_input(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """NO-OP: Skip initial split (Point 1).
        
        API: patches (Injection)
        Hook: patches["post_input"] (model.py line 127)
        
        DESIGN DECISION: ComfyUI's patches API doesn't reliably pass 
        transformer_options to post_input hooks. Rather than fight the API,
        we let double_blocks process the FULL sequence on each GPU.
        
        This is ACCEPTABLE because:
        - FSDP2 model can handle full sequence (just uses more VRAM)
        - Peak: ~15GB vs 11GB baseline (within 24GB capacity)
        - Gather/resplit chain still works correctly
        - Simpler than threading transformer_options through closure hacks
        
        The actual Ring-Attention split happens at:
        - patch_last_double_block: Gather full→full, resplit for single_blocks
        - patch_last_single_block: Final gather and img extraction
        
        Args:
            args: {"img": tensor, "txt": tensor, "img_ids": tensor, ...}
        
        Returns:
            args dict unchanged (no split)
        """
        # NO-OP: Split happens in patch_first_double_block instead
        # patches["post_input"] doesn't reliably receive transformer_options
        return args
    
    # ========================================================================
    # POINT 2: Gather/Resplit (patches_replace API - Chained Override)
    # ========================================================================
    
    def patch_last_double_block(
        self, 
        args: Dict[str, Any], 
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gather after last double_block, concat, resplit (Point 2 - Part 1).
        
        API: patches_replace (Override)
        Hook: patches_replace["dit"][("double_block", N-1)] (model.py line 142)
        
        This function wraps the LAST double_block to:
        1. Execute the original block with local tensors
        2. All-gather img and txt from all ranks
        3. Concatenate (txt_full, img_full) as Flux expects
        4. Re-split the combined tensor for single_blocks
        5. "Poison" the return with None to bypass line 182's torch.cat
        6. Pass the correct tensor via transformer_options to the next patch
        
        Args:
            args: {
                "img": [batch, local_seq, hidden],
                "txt": [batch, local_txt_len, hidden],
                "vec": embeddings,
                "pe": positional embeddings,
                "transformer_options": state bag
            }
            objects: {
                "original_block": callable wrapper to original block
            }
        
        Returns:
            {"img": None, "txt": None}  # Poisoned to bypass line 182
        """
        # Get ring context
        transformer_options = args.get("transformer_options", {})
        ring_context = transformer_options.get("ring_context")
        
        if ring_context is None:
            # Fallback: just execute original block
            return objects["original_block"](args)
        
        rank = ring_context["rank"]
        world_size = ring_context["world_size"]
        sp_group = ring_context["sp_group"]
        
        # CRITICAL: Replace full pe with split pe (same as middle blocks)
        ring_split_pe = transformer_options.get("ring_split_pe")
        if ring_split_pe is not None:
            logging.info(
                f"{LOG_PREFIX}[Rank {rank}] Last double_block: replacing pe {args['pe'].shape} with split {ring_split_pe.shape}"
            )
            args["pe"] = ring_split_pe
        else:
            logging.warning(
                f"{LOG_PREFIX}[Rank {rank}] Last double_block: ring_split_pe NOT FOUND!"
            )
        
        logging.debug(
            f"{LOG_PREFIX}[Rank {rank}] Last double_block: "
            f"executing original with local tensors"
        )
        
        # 1. Execute original block with local tensors
        out = objects["original_block"](args)
        
        img_local = out["img"]
        txt_local = out["txt"]
        
        logging.debug(
            f"{LOG_PREFIX}[Rank {rank}] Last double_block output: "
            f"img={img_local.shape}, txt={txt_local.shape}"
        )
        
        # 2. All-gather from all ranks
        logging.info(f"{LOG_PREFIX}[Rank {rank}] ✅ LAST DOUBLE_BLOCK PATCH: Gathering img and txt from all ranks")
        
        img_full = all_gather_sequence(img_local, sp_group, dim=1)
        txt_full = all_gather_sequence(txt_local, sp_group, dim=1)
        
        logging.info(f"{LOG_PREFIX}[Rank {rank}] Gathered: img {img_local.shape} → {img_full.shape}, txt {txt_local.shape} → {txt_full.shape}")
        
        # 3. Store txt_length for final extraction (Point 3)
        ring_context["txt_length"] = txt_full.shape[1]
        
        logging.debug(
            f"{LOG_PREFIX}[Rank {rank}] Gathered: "
            f"img={img_full.shape}, txt={txt_full.shape}"
        )
        
        # 4. Concatenate as Flux expects (txt first, then img)
        combined = torch.cat((txt_full, img_full), dim=1)
        
        # 5. Re-split for single_blocks
        combined_local = torch.chunk(combined, world_size, dim=1)[rank].contiguous()
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] Resplit for single_blocks: "
            f"combined {combined.shape} → {combined_local.shape}"
        )
        
        # 6. Pass the correct tensor to the next patch via transformer_options
        # This bridges the "Critical Gap" at line 182
        ring_context["pre_single_block_tensor"] = combined_local
        
        # 7. Return dummy tensors to bypass line 182's torch.cat
        # Line 182 will concat these, but result is ignored by patch_first_single_block
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] CRITICAL: Returning tensors for line 182 bypass"
        )
        
        # Create dummy tensors with CORRECT local size so line 182 works
        # Line 182: img = torch.cat((txt, img), 1)
        # Must match combined_local size (which is what patch_first_single_block expects)
        batch_size = combined_local.shape[0]
        local_seq_len = combined_local.shape[1]
        hidden_dim = combined_local.shape[2]
        device = combined_local.device
        dtype = combined_local.dtype
        
        # Return the ACTUAL combined_local split between img/txt
        # Doesn't matter how we split since patch_first_single_block replaces it
        dummy_img = combined_local  # Just return the full thing
        dummy_txt = torch.zeros((batch_size, 0, hidden_dim), device=device, dtype=dtype)  # Empty
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] ✅ RETURNING: img={dummy_img.shape}, txt={dummy_txt.shape}"
        )
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] Line 182 will concat these: txt {dummy_txt.shape} + img {dummy_img.shape}"
        )
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] Expected result: torch.Size([{batch_size}, {local_seq_len}, {hidden_dim}])"
        )
        
        return {"img": dummy_img, "txt": dummy_txt}
    
    def patch_first_single_block(
        self, 
        args: Dict[str, Any], 
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Catch bridged tensor and execute first single_block (Point 2 - Part 2).
        
        API: patches_replace (Override)
        Hook: patches_replace["dit"][("single_block", 0)] (model.py line 190)
        
        This function completes the "Chained Patch" by:
        1. Retrieving the correct combined_local tensor from transformer_options
        2. Replacing args["img"] (which contains the poisoned result from line 182)
        3. Executing the original block with the CORRECT tensor
        
        Args:
            args: {
                "img": <POISONED> result from line 182 (IGNORED),
                "vec": embeddings,
                "pe": positional embeddings,
                "transformer_options": state bag
            }
            objects: {
                "original_block": callable wrapper to original block
            }
        
        Returns:
            {"img": output from original block with correct input}
        """
        # Get ring context
        transformer_options = args.get("transformer_options", {})
        ring_context = transformer_options.get("ring_context")
        
        if ring_context is None:
            # Fallback: execute original block
            return objects["original_block"](args)
        
        rank = ring_context["rank"]
        
        # 1. Retrieve the bridged tensor
        combined_local = ring_context.get("pre_single_block_tensor")
        
        if combined_local is None:
            raise RuntimeError(
                f"{LOG_PREFIX}[Rank {rank}] ❌ Chained Patch BROKEN: "
                f"pre_single_block_tensor is None. "
                f"patch_last_double_block did not execute correctly."
            )
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] ✅ FIRST SINGLE_BLOCK PATCH: Caught bridged tensor: "
            f"shape={combined_local.shape}"
        )
        
        # 2. CRITICAL: Replace args["img"] with the CORRECT tensor
        # args["img"] currently contains the poisoned result from line 182
        poisoned_img = args["img"]
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] BEFORE REPLACEMENT: args['img'].shape = {poisoned_img.shape}"
        )
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] REPLACEMENT: combined_local.shape = {combined_local.shape}"
        )
        
        args["img"] = combined_local
        
        # CRITICAL: Also replace pe with split version (same as all other blocks!)
        transformer_options = args.get("transformer_options", {})
        ring_split_pe = transformer_options.get("ring_split_pe")
        if ring_split_pe is not None:
            logging.info(
                f"{LOG_PREFIX}[Rank {rank}] ALSO replacing pe {args['pe'].shape} with split {ring_split_pe.shape}"
            )
            args["pe"] = ring_split_pe
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] AFTER REPLACEMENT: args['img'].shape = {args['img'].shape}, pe={args['pe'].shape}"
        )
        
        # 3. Clear the context variable to free memory
        ring_context["pre_single_block_tensor"] = None
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] ✅ Injected correct tensor, executing original single_block[0]"
        )
        
        # 4. Execute original block with correct input
        return objects["original_block"](args)
    
    def patch_middle_single_block(
        self,
        args: Dict[str, Any],
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use split pe for middle single_blocks (1-36).
        
        Same as patch_middle_double_block but for single_blocks.
        """
        transformer_options = args.get("transformer_options", {})
        ring_split_pe = transformer_options.get("ring_split_pe")
        ring_context = transformer_options.get("ring_context", {})
        rank = ring_context.get("rank", "?")
        
        logging.info(f"{LOG_PREFIX}[Rank {rank}] Middle single_block: ring_split_pe={'PRESENT' if ring_split_pe is not None else 'MISSING'}")
        
        if ring_split_pe is not None:
            args["pe"] = ring_split_pe
            # Skip detailed logging - just mark as replaced
        else:
            logging.warning(f"{LOG_PREFIX}[Rank {rank}] Middle single_block: NO ring_split_pe!")
        
        return objects["original_block"](args)
    
    # ========================================================================
    # POINT 3: Final Gather (patches_replace API - Override)
    # ========================================================================
    
    def patch_last_single_block(
        self, 
        args: Dict[str, Any], 
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute last single_block and perform final gather (Point 3).
        
        API: patches_replace (Override)
        Hook: patches_replace["dit"][("single_block", M-1)] (model.py line 190)
        
        This function wraps the LAST single_block to:
        1. Execute the original block with local combined tensor
        2. All-gather the output from all ranks
        3. Extract the img portion (skipping the txt prefix)
        4. Return the full, reconstructed output
        
        Args:
            args: {
                "img": [batch, local_combined_seq, hidden],
                "vec": embeddings,
                "pe": positional embeddings,
                "transformer_options": state bag
            }
            objects: {
                "original_block": callable wrapper to original block
            }
        
        Returns:
            {"img": [batch, full_img_seq, hidden]}
        """
        # Get ring context
        transformer_options = args.get("transformer_options", {})
        ring_context = transformer_options.get("ring_context")
        
        if ring_context is None:
            # Fallback: execute original block
            return objects["original_block"](args)
        
        rank = ring_context["rank"]
        sp_group = ring_context["sp_group"]
        txt_length = ring_context.get("txt_length")
        
        if txt_length is None:
            raise RuntimeError(
                f"{LOG_PREFIX}[Rank {rank}] Chained Patch BROKEN: "
                f"txt_length was not set by patch_last_double_block."
            )
        
        # CRITICAL: Replace full pe with split pe (same as all other blocks!)
        ring_split_pe = transformer_options.get("ring_split_pe")
        if ring_split_pe is not None:
            logging.info(
                f"{LOG_PREFIX}[Rank {rank}] Last single_block: replacing pe with split"
            )
            args["pe"] = ring_split_pe
        else:
            logging.warning(
                f"{LOG_PREFIX}[Rank {rank}] Last single_block: ring_split_pe NOT FOUND!"
            )
        
        logging.debug(
            f"{LOG_PREFIX}[Rank {rank}] Last single_block: "
            f"executing original with local tensor"
        )
        
        # 1. Execute original block
        out = objects["original_block"](args)
        output_local = out["img"]
        
        logging.debug(
            f"{LOG_PREFIX}[Rank {rank}] Last single_block output: "
            f"shape={output_local.shape}"
        )
        
        # 2. All-gather final output
        logging.info(f"{LOG_PREFIX}[Rank {rank}] ✅ LAST SINGLE_BLOCK PATCH: Final gather from all ranks")
        
        output_full = all_gather_sequence(output_local, sp_group, dim=1)
        
        logging.info(f"{LOG_PREFIX}[Rank {rank}] Final gathered: {output_local.shape} → {output_full.shape}")
        
        logging.debug(
            f"{LOG_PREFIX}[Rank {rank}] Final gathered: "
            f"shape={output_full.shape}"
        )
        
        # 3. Extract img portion (skip txt prefix)
        output_img = output_full[:, txt_length:, ...].contiguous()
        
        logging.info(
            f"{LOG_PREFIX}[Rank {rank}] Extracted img: "
            f"{output_full.shape} → {output_img.shape} "
            f"(removed {txt_length} txt tokens)"
        )
        
        # 4. Return full reconstructed output
        # This will be passed to self.final_layer (model.py line 205)
        return {"img": output_img}
