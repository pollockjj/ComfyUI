"""Distributed Model Wrapper for FSDP models in worker processes.

Implements ComfyUI's model interface (BaseModel methods) and forwards
apply_model() calls to workers via MultiprocExecutor. Workers execute
forward pass with FSDP-sharded model and return outputs.

This allows ComfyUI samplers to work unchanged while actual model
lives in distributed worker processes.

Design Philosophy: EXTEND ComfyUI Interface
- Implements apply_model(), get_dtype(), get_model_object()
- Forwards calls via executor.execute_collective()
- Stateless proxy (no model weights, only executor reference)
- Transparent to ComfyUI samplers

Based on:
- Phase 4 DistributedModelWrapper pattern
- Raylight integration analysis (RPC model)
- FastVideo collective execution model
"""

from __future__ import annotations
import torch
import logging
from typing import Optional, Dict, Any

LOG_PREFIX = "⚡ [Parallel-Attention]"


class DistributedModelWrapper:
    """Proxy for FSDP models living in worker processes.
    
    Implements ComfyUI's model interface and forwards apply_model() calls
    to workers via MultiprocExecutor. Workers execute forward pass with
    FSDP-sharded model and return results.
    
    Attributes:
        executor: MultiprocExecutor managing worker processes
        model_config: Model configuration from workers
        dtype: Model dtype for ComfyUI interface
        model_type: Model architecture name (flux, wan, etc.)
        is_fsdp: Whether model is FSDP-wrapped
        world_size: Number of worker processes
    
    Example:
        # Created by loader after model loaded in workers
        wrapper = DistributedModelWrapper(
            executor=executor,
            model_config={
                "model_type": "flux",
                "dtype": "bfloat16",
                "is_fsdp": True,
                "world_size": 2
            }
        )
        
        # Use exactly like ComfyUI model
        output = wrapper.apply_model(x, t, c_crossattn=context)
    """
    
    def __init__(self, executor, scaffold: Dict[str, Any]):
        """Initialize distributed model wrapper with complete scaffold.
        
        Uses "Copy at Perfect Information" pattern - receives complete
        model metadata extracted during CPU/meta load, eliminating
        piecemeal property discovery.
        
        Args:
            executor: MultiprocExecutor with model loaded in workers
            scaffold: Complete model metadata from extract_model_scaffold():
                - model_type: Architecture name (flux, wan, etc.)
                - dtype: Model dtype ('bfloat16', 'float16', etc.)
                - latent_format: Serialized latent format object
                - is_adm: ADM conditioning flag
                - extra_conds: Extra conditioning dict
                - model_options: Model-specific options
                - load_device: Target device string
                - offload_device: CPU offload device string
                - model_size: Total unsharded model size
        """
        self.executor = executor
        self.scaffold = scaffold
        
        # Deserialize properties from scaffold (perfect information)
        dtype_str = scaffold.get('dtype', 'bfloat16')
        self.dtype = getattr(torch, dtype_str)
        
        self.model_type = scaffold.get('model_type', 'unknown')
        self.is_fsdp = True  # Always FSDP if using distributed wrapper
        self.world_size = 2  # Will be passed in scaffold later
        
        # Deserialize latent_format (THE KEY FIX)
        from comfy.parallel_attention.model_scaffold import deserialize_latent_format
        self._latent_format = deserialize_latent_format(scaffold["latent_format"])
        
        # Conditioning properties
        self._is_adm = scaffold.get("is_adm", False)
        self.extra_conds_dict = scaffold.get("extra_conds", {})
        
        # ComfyUI compatibility attributes
        self.load_device = torch.device(scaffold.get("load_device", "cuda:0"))
        self.offload_device = torch.device(scaffold.get("offload_device", "cpu"))
        self.model_options = scaffold.get("model_options", {})
        
        # Memory properties from scaffold
        self._model_size = scaffold.get("model_size", 20 * 1024**3)
        
        logging.info(
            f"{LOG_PREFIX} [Wrapper] Created from scaffold: "
            f"type={self.model_type}, dtype={dtype_str}, "
            f"latent_format={self._latent_format.__class__.__name__}, "
            f"size={self._model_size / (1024**3):.2f}GB"
        )
    
    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: Optional[torch.Tensor] = None,
        c_crossattn: Optional[torch.Tensor] = None,
        control=None,
        transformer_options: Optional[Dict] = None,
        **kwargs
    ) -> torch.Tensor:
        """Execute forward pass in workers via RPC.
        
        Called by ComfyUI samplers during denoising loop. Forwards inputs
        to workers, blocks until forward complete, returns output to main.
        
        Args:
            x: Latent tensor [B, C, H, W]
            t: Timesteps tensor [B] or scalar
            c_concat: Optional concat conditioning
            c_crossattn: Text conditioning (context) [B, seq_len, dim]
            control: ControlNet conditioning (not yet supported)
            transformer_options: Model-specific options dict
            **kwargs: Additional model-specific arguments
        
        Returns:
            torch.Tensor: Denoised output [B, C, H, W]
        
        Raises:
            RuntimeError: If forward pass fails in workers
        """
        if transformer_options is None:
            transformer_options = {}
        
        # Pack inputs for serialization to workers
        # Move to CPU for pipe serialization
        inputs = {
            "x": x.cpu() if x.device.type != 'cpu' else x,
            "t": t.cpu() if torch.is_tensor(t) and t.device.type != 'cpu' else t,
            "c_concat": c_concat.cpu() if c_concat is not None and c_concat.device.type != 'cpu' else c_concat,
            "c_crossattn": c_crossattn.cpu() if c_crossattn is not None and c_crossattn.device.type != 'cpu' else c_crossattn,
            "control": control,
            "transformer_options": transformer_options,
            **kwargs
        }
        
        # Execute collectively (all workers process identical input)
        # Workers will:
        # 1. Receive inputs (broadcast to all ranks)
        # 2. Execute model.apply_model() with FSDP all-gather
        # 3. Return output (gathered from rank 0)
        result = self.executor.execute_collective("forward_pass", inputs)
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(
                f"{LOG_PREFIX} [Wrapper] Forward pass failed in workers: "
                f"{result['error']}"
            )
        
        # Extract output tensor
        output = result.get("output") if isinstance(result, dict) else result
        
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                f"{LOG_PREFIX} [Wrapper] Invalid output type: {type(output)}"
            )
        
        # Move back to nominal device for ComfyUI
        return output.to(self.load_device)
    
    def get_dtype(self) -> torch.dtype:
        """Return model dtype for ComfyUI casting.
        
        ComfyUI uses this to cast inputs to correct dtype.
        
        Returns:
            torch.dtype: Model's compute dtype
        """
        return self.dtype
    
    def get_model_object(self, name: str):
        """Return model object by name for ComfyUI interface.
        
        Returns objects from scaffold (no RPC needed - closed loop).
        This is where scaffold pattern shines: all metadata already
        available from perfect information extraction.
        
        Args:
            name: Model component name
        
        Returns:
            Requested object from scaffold or None
        """
        if name == "latent_format":
            # Return cached latent_format from scaffold
            return self._latent_format
        elif name in ("model", "diffusion_model", ""):
            return self
        return None
    
    def model_dtype(self) -> torch.dtype:
        """Alias for get_dtype() for ComfyUI compatibility.
        
        Returns:
            torch.dtype: Model's compute dtype
        """
        return self.dtype
    
    def is_adm(self) -> bool:
        """Check if model uses ADM conditioning.
        
        Returns from scaffold (no RPC needed).
        
        Returns:
            bool: Whether model uses ADM (Stable Diffusion style)
        """
        return self._is_adm
    
    def encode_adm(self, **kwargs):
        """Encode ADM conditioning (if supported).
        
        Args:
            **kwargs: ADM encoding parameters
        
        Returns:
            ADM encoding tensor or None
        """
        if not self.is_adm():
            return None
        
        # Forward ADM encoding to workers if model supports it
        result = self.executor.execute_collective("encode_adm", kwargs)
        return result.get("adm_encoding", None)
    
    def extra_conds(self, **kwargs):
        """Return extra conditioning requirements.
        
        Returns from scaffold (no RPC needed).
        
        Returns:
            dict: Extra conditioning configuration
        """
        return self.extra_conds_dict
    
    def model_size(self):
        """Return total model size for memory scheduler.
        
        Returns unsharded model size (e.g., 22GB for Flux).
        Memory scheduler uses this for loading decisions.
        
        Returns:
            int: Model size in bytes (unsharded)
        """
        # Already extracted from scaffold in __init__ (line 88)
        return self._model_size
    
    def loaded_size(self):
        """Return wrapper's memory footprint.
        
        Wrapper is just a proxy object (~1KB). Actual model
        memory is in worker processes (reported separately).
        
        Returns:
            int: Wrapper overhead in bytes (~1KB)
        """
        return 1024  # 1KB wrapper overhead
    
    def model_memory_required(self, input_shape):
        """Estimate memory required for forward pass.
        
        This is approximate - actual memory is in workers.
        Memory scheduler uses this to decide if forward pass fits.
        
        Args:
            input_shape: Input tensor shape tuple
        
        Returns:
            int: Estimated memory in bytes
        """
        # Wrapper has minimal overhead
        # Real memory (FSDP shards + activations) is in workers
        # Return conservative estimate based on input size
        
        import numpy as np
        
        # Calculate activation memory (rough estimate)
        # Assume fp16 (2 bytes per element)
        input_elements = int(np.prod(input_shape))
        activation_memory = input_elements * 2  # fp16
        
        # Add some overhead for intermediate tensors
        estimated_memory = activation_memory * 3  # 3x for intermediates
        
        return estimated_memory
    
    def __repr__(self) -> str:
        """String representation for debugging.
        
        Returns:
            str: Human-readable representation
        """
        return (
            f"DistributedModelWrapper("
            f"type={self.model_type}, "
            f"fsdp={self.is_fsdp}, "
            f"workers={self.world_size})"
        )
