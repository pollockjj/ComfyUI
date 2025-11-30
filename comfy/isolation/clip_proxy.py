"""
Stateless RPC pattern for CLIP instances.

This module provides:
1. CLIPRegistry - Host-side registry of CLIP instances (ProxiedSingleton)
2. CLIPProxy - Picklable handle that forwards calls via RPC

Architecture mirrors model_sampling_proxy.py exactly.
"""

import asyncio
import logging
import os
import pickle
import threading
import weakref
from typing import Any, Dict, Optional

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    # Graceful degradation if pyisolate not available
    class ProxiedSingleton:
        """Fallback when pyisolate not installed."""
        pass

logger = logging.getLogger(__name__)

# Host/child detection
IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"


class CLIPRegistry(ProxiedSingleton):
    """
    Host-side registry of CLIP instances using ProxiedSingleton pattern.
    
    Thread-safe singleton that manages CLIP object lifecycle and provides
    async RPC methods for isolated child processes.
    
    CRITICAL: Inherits from ProxiedSingleton to enable RPC from child processes.
    """
    
    def __init__(self) -> None:
        """Initialize registry state (called once by ProxiedSingleton)."""
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        
        self._registry: Dict[str, Any] = {}
        self._id_map: Dict[int, str] = {}  # id(clip) → instance_id (identity preservation)
        self._counter = 0
        self._lock = threading.Lock()
        logger.debug("📚 [PyIsolate][CLIPRegistry] Initialized")
    
    def register(self, clip_instance) -> str:
        """
        Register a CLIP instance and return unique ID.
        
        If the same Python object (by id()) was already registered,
        returns the existing ID to preserve identity semantics.
        
        Args:
            clip_instance: CLIP object to register
            
        Returns:
            Unique instance ID (e.g., "clip_0")
            
        Raises:
            RuntimeError: If called from child process
        """
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                "📚 [PyIsolate][CLIPRegistry] FAIL-LOUD: "
                "Cannot register CLIP in child process"
            )
        
        with self._lock:
            # Check if already registered (identity preservation)
            obj_id = id(clip_instance)
            if obj_id in self._id_map:
                existing_id = self._id_map[obj_id]
                logger.debug(
                    f"📚 [PyIsolate][CLIPRegistry] Re-using {existing_id} for object {obj_id}"
                )
                return existing_id
            
            # New registration
            instance_id = f"clip_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = clip_instance
            self._id_map[obj_id] = instance_id
            logger.debug(
                f"📚 [PyIsolate][CLIPRegistry] Registered {instance_id}"
            )
        
        return instance_id
    
    def unregister_sync(self, instance_id: str) -> None:
        """
        Unregister a CLIP instance (called by weakref.finalize).
        
        This is a synchronous method designed to be called from finalizers.
        Must not raise exceptions.
        
        Args:
            instance_id: ID to unregister
        """
        try:
            with self._lock:
                if instance_id in self._registry:
                    del self._registry[instance_id]
                    logger.debug(
                        f"📚 [PyIsolate][CLIPRegistry] Unregistered {instance_id}"
                    )
        except Exception as e:
            logger.error(
                f"📚 [PyIsolate][CLIPRegistry] Unregister failed for {instance_id}: {e}"
            )
    
    def _get_instance(self, instance_id: str):
        """
        Internal: Get CLIP instance by ID.
        
        Args:
            instance_id: ID to lookup
            
        Returns:
            CLIP instance
            
        Raises:
            ValueError: If instance_id not found
        """
        instance = self._registry.get(instance_id)
        if instance is None:
            raise ValueError(
                f"📚 [PyIsolate][CLIPRegistry] FAIL-LOUD: "
                f"Instance {instance_id} not found in registry"
            )
        return instance
    
    # RPC Methods (async) - Simple Operations
    
    async def get_ram_usage(self, instance_id: str) -> int:
        """RPC: Get RAM usage of CLIP instance."""
        instance = self._get_instance(instance_id)
        return instance.get_ram_usage()
    
    async def clip_layer(self, instance_id: str, layer_idx: int) -> None:
        """RPC: Set CLIP layer index."""
        instance = self._get_instance(instance_id)
        instance.clip_layer(layer_idx)
    
    async def set_tokenizer_option(
        self, instance_id: str, option_name: str, value: Any
    ) -> None:
        """RPC: Set tokenizer option."""
        instance = self._get_instance(instance_id)
        instance.set_tokenizer_option(option_name, value)
    
    # RPC Methods (async) - Tokenization
    
    async def tokenize(
        self,
        instance_id: str,
        text: str,
        return_word_ids: bool = False,
        **kwargs
    ) -> dict:
        """RPC: Tokenize text."""
        instance = self._get_instance(instance_id)
        return instance.tokenize(text, return_word_ids=return_word_ids, **kwargs)
    
    # RPC Methods (async) - Encoding (Returns Tensors)
    
    async def encode(self, instance_id: str, text: str):
        """RPC: Encode text to embeddings (returns tensor)."""
        instance = self._get_instance(instance_id)
        result = instance.encode(text)
        # Tensors are automatically handled by share_torch=true
        return result
    
    async def encode_from_tokens(
        self,
        instance_id: str,
        tokens: Any,
        return_pooled: bool = False,
        return_dict: bool = False
    ):
        """RPC: Encode from tokens (returns tensor/tuple/dict)."""
        instance = self._get_instance(instance_id)
        result = instance.encode_from_tokens(
            tokens, return_pooled=return_pooled, return_dict=return_dict
        )
        # share_torch=true handles tensor returns automatically
        return result
    
    async def encode_from_tokens_scheduled(
        self,
        instance_id: str,
        tokens: Any,
        unprojected: bool = False,
        add_dict: dict = None,
        show_pbar: bool = True
    ):
        """RPC: Scheduled encoding (returns list of tuples with tensors)."""
        instance = self._get_instance(instance_id)
        if add_dict is None:
            add_dict = {}
        result = instance.encode_from_tokens_scheduled(
            tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar
        )
        # share_torch=true handles nested tensors in lists/tuples
        return result
    
    # RPC Methods (async) - Clone (Deep Remote Copy)
    
    async def clone(self, instance_id: str) -> str:
        """
        RPC: Clone CLIP instance (Deep Remote Copy pattern).
        
        Creates a new CLIP instance via clone(), registers it,
        and returns the new ID.
        
        Args:
            instance_id: Source CLIP ID
            
        Returns:
            New instance ID for the clone
        """
        instance = self._get_instance(instance_id)
        new_clip = instance.clone()
        new_id = self.register(new_clip)
        logger.debug(
            f"📚 [PyIsolate][CLIPRegistry] Cloned {instance_id} → {new_id}"
        )
        return new_id
    
    # RPC Methods (async) - LoRA/Patching
    
    async def add_patches(
        self,
        instance_id: str,
        patches: Any,
        strength_patch: float = 1.0,
        strength_model: float = 1.0
    ):
        """RPC: Add patches (LoRA)."""
        instance = self._get_instance(instance_id)
        return instance.add_patches(
            patches, strength_patch=strength_patch, strength_model=strength_model
        )
    
    async def get_key_patches(self, instance_id: str) -> dict:
        """RPC: Get key patches."""
        instance = self._get_instance(instance_id)
        return instance.get_key_patches()
    
    # RPC Methods (async) - State Dict
    
    async def load_sd(
        self, instance_id: str, sd: dict, full_model: bool = False
    ):
        """RPC: Load state dict."""
        instance = self._get_instance(instance_id)
        return instance.load_sd(sd, full_model=full_model)
    
    async def get_sd(self, instance_id: str) -> dict:
        """RPC: Get state dict."""
        instance = self._get_instance(instance_id)
        return instance.get_sd()


class CLIPProxy:
    """
    Lightweight, picklable handle to a CLIP instance.
    
    Design Principles:
    1. Zero State: Only stores instance_id + registry reference
    2. Host Optimization: Bypasses RPC when running on host (_is_child=False)
    3. Transparent: Appears identical to CLIP from node's perspective
    4. Fail-Loud: Any RPC failure raises immediately (no silent failures)
    """
    
    def __init__(
        self,
        instance_id: str,
        registry: Optional[CLIPRegistry] = None,
        manage_lifecycle: bool = False
    ):
        """
        Initialize CLIPProxy.
        
        Args:
            instance_id: Registry ID of the CLIP instance
            registry: CLIPRegistry singleton (auto-created if None)
            manage_lifecycle: If True, proxy manages cleanup via weakref.finalize
        """
        self._instance_id = instance_id
        self._manage_lifecycle = manage_lifecycle
        self._is_child = os.environ.get("PYISOLATE_CHILD") == "1"
        
        # Registry passed in explicitly (from deserialization or manual creation)
        self._registry = registry if registry is not None else CLIPRegistry()
        
        # Lifecycle: only host-side proxy cleans up
        if manage_lifecycle and not self._is_child:
            self._finalizer = weakref.finalize(
                self, self._registry.unregister_sync, instance_id
            )
            logger.debug(
                f"📚 [PyIsolate][CLIPProxy] Lifecycle management enabled for {instance_id}"
            )
    
    def __reduce__(self):
        """
        Custom pickle - only serialize instance_id.
        
        Returns tuple for pickle reconstruction with is_new_object=False
        to prevent double-finalize on round-trip.
        """
        return (_reconstruct_clip_proxy, (self._instance_id, False))
    
    def _call_registry(self, method_name: str, *args, **kwargs):
        """
        Call registry method with host-side optimization.
        
        If running on host: call directly (no RPC overhead)
        If running in child: use RpcBridge for sync-to-async
        
        Args:
            method_name: Name of CLIPRegistry method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from registry method
        """
        # Get registry dynamically (important for child processes)
        if self._registry is None:
            self._registry = CLIPRegistry()
        
        method = getattr(self._registry, method_name)
        
        if self._is_child:
            # Child process: RPC to host via ProxiedSingleton mechanism
            # The registry instance will automatically be the remote proxy
            from comfy.isolation.rpc_bridge import RpcBridge
            bridge = RpcBridge()
            return bridge.run_sync(method(self._instance_id, *args, **kwargs))
        else:
            # Host process: direct call (CRITICAL OPTIMIZATION)
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                method(self._instance_id, *args, **kwargs)
            )
    
    # Simple Methods
    
    def get_ram_usage(self) -> int:
        """Get RAM usage of CLIP instance."""
        return self._call_registry('get_ram_usage')
    
    def clip_layer(self, layer_idx: int) -> None:
        """Set CLIP layer index."""
        return self._call_registry('clip_layer', layer_idx)
    
    def set_tokenizer_option(self, option_name: str, value: Any) -> None:
        """Set tokenizer option."""
        return self._call_registry('set_tokenizer_option', option_name, value)
    
    # Tokenization
    
    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs) -> dict:
        """Tokenize text."""
        return self._call_registry('tokenize', text, return_word_ids, **kwargs)
    
    # Encoding Methods (Return Tensors)
    
    def encode(self, text: str):
        """Encode text to embeddings."""
        return self._call_registry('encode', text)
    
    def encode_from_tokens(
        self, tokens: Any, return_pooled: bool = False, return_dict: bool = False
    ):
        """Encode from tokens."""
        return self._call_registry(
            'encode_from_tokens', tokens, return_pooled, return_dict
        )
    
    def encode_from_tokens_scheduled(
        self,
        tokens: Any,
        unprojected: bool = False,
        add_dict: dict = None,
        show_pbar: bool = True
    ):
        """Scheduled encoding."""
        if add_dict is None:
            add_dict = {}
        return self._call_registry(
            'encode_from_tokens_scheduled', tokens, unprojected, add_dict, show_pbar
        )
    
    # Clone (Deep Remote Copy)
    
    def clone(self) -> 'CLIPProxy':
        """
        Clone CLIP instance (Deep Remote Copy pattern).
        
        Creates a new CLIP instance in the registry and returns a new proxy.
        Host-side clones manage lifecycle; child-side do not.
        
        Returns:
            New CLIPProxy for the cloned instance
        """
        new_id = self._call_registry('clone')
        # Host-side clones manage lifecycle; child-side do not
        return CLIPProxy(new_id, self._registry, manage_lifecycle=not self._is_child)
    
    # LoRA/Patching
    
    def add_patches(
        self, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0
    ):
        """Add patches (LoRA)."""
        return self._call_registry(
            'add_patches', patches, strength_patch, strength_model
        )
    
    def get_key_patches(self) -> dict:
        """Get key patches."""
        return self._call_registry('get_key_patches')
    
    # State Dict
    
    def load_sd(self, sd: dict, full_model: bool = False):
        """Load state dict."""
        return self._call_registry('load_sd', sd, full_model)
    
    def get_sd(self) -> dict:
        """Get state dict."""
        return self._call_registry('get_sd')
    
    # load_model (SCOPED OUT for Phase 1)
    
    def load_model(self):
        """SCOPED OUT for Phase 1 - ModelPatcher cannot be proxied yet."""
        raise NotImplementedError(
            "load_model() is not supported in isolated mode. "
            "Access model functionality via CLIP methods (encode, tokenize, etc.)."
        )
    
    # Property Guards (Raise AttributeError)
    
    @property
    def patcher(self):
        """Property guard: patcher access not supported."""
        raise AttributeError(
            "Direct access to 'patcher' is not supported in isolated mode. "
            "Use add_patches() and get_key_patches() instead."
        )
    
    @property
    def cond_stage_model(self):
        """Property guard: cond_stage_model access not supported."""
        raise AttributeError(
            "Direct access to 'cond_stage_model' is not supported in isolated mode. "
            "Use CLIP methods (encode, tokenize) instead."
        )
    
    @property
    def layer_idx(self):
        """Property guard: layer_idx access not supported."""
        raise AttributeError(
            "Direct access to 'layer_idx' is not supported in isolated mode. "
            "Use clip_layer() to set layer index."
        )
    
    @property
    def use_clip_schedule(self):
        """Property guard: use_clip_schedule access not supported."""
        raise AttributeError(
            "Direct access to 'use_clip_schedule' is not supported in isolated mode."
        )


def _reconstruct_clip_proxy(clip_id: str, is_new_object: bool = True) -> CLIPProxy:
    """
    Pickle reconstruction helper.
    
    Args:
        clip_id: Registry ID of the CLIP instance
        is_new_object: True if this is a NEW object (e.g., clone result),
                       False if this is a round-trip of existing proxy.
                       
    Lifecycle Rules:
        - Child process: NEVER manage lifecycle (always False)
        - Host process, new object: manage lifecycle (True)
        - Host process, round-trip: do NOT manage (False) - original proxy owns it
    
    Returns:
        Reconstructed CLIPProxy
    """
    IS_CHILD = os.environ.get("PYISOLATE_CHILD") == "1"
    # Don't instantiate CLIPRegistry() here - let CLIPProxy._call_registry do it lazily
    # This prevents "Cannot inject instance after first instantiation" errors
    # when use_remote() hasn't been called yet in child processes
    registry = None
    
    if IS_CHILD:
        # Child never manages lifecycle
        return CLIPProxy(clip_id, registry, manage_lifecycle=False)
    else:
        # Host: manage only if this is a new object (clone, etc.)
        return CLIPProxy(clip_id, registry, manage_lifecycle=is_new_object)


def maybe_wrap_clip_for_isolation(clip):
    """
    Wrap CLIP in isolation proxy if isolation is active.
    
    Called from checkpoint loading path.
    Returns original clip if:
    - Isolation not active
    - Already in child process
    - Already a CLIPProxy
    
    Args:
        clip: CLIP instance to potentially wrap
        
    Returns:
        CLIPProxy if isolation active, otherwise original clip
    """
    if os.environ.get("PYISOLATE_ISOLATION_ACTIVE") != "1":
        return clip
    if os.environ.get("PYISOLATE_CHILD") == "1":
        return clip
    if isinstance(clip, CLIPProxy):
        return clip
    
    registry = CLIPRegistry()
    clip_id = registry.register(clip)
    logger.debug(f"📚 [PyIsolate][CLIPProxy] Wrapped CLIP as {clip_id}")
    return CLIPProxy(clip_id, registry, manage_lifecycle=True)
