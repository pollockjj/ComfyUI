"""Capture and validate against golden reference from working serial implementation."""

import logging
import torch
import hashlib
import json
from pathlib import Path

_logger = logging.getLogger(__name__)

GOLDEN_DIR = Path("/home/johnj/split-q/golden_activations")
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def tensor_hash(tensor):
    """Compute SHA256 hash of tensor for byte-exact comparison."""
    if tensor is None:
        return "none"
    # Move to CPU, ensure contiguous, get bytes
    t_cpu = tensor.detach().cpu().contiguous()
    data_bytes = t_cpu.numpy().tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


def save_golden_step(step_idx, latent, label="latent", metadata=None):
    """Save golden reference for a single step with label."""
    golden_file = GOLDEN_DIR / f"step_{step_idx:03d}_{label}.json"
    
    data = {
        "step": step_idx,
        "label": label,
        "latent_hash": tensor_hash(latent),
        "latent_shape": list(latent.shape),
        "latent_device": str(latent.device),
        "latent_dtype": str(latent.dtype),
        "metadata": metadata or {}
    }
    
    golden_file.write_text(json.dumps(data, indent=2))
    _logger.info(f"⚡ [split-q][Golden] Saved step {step_idx} {label}: hash={data['latent_hash']}")
    
    return data


def load_golden_step(step_idx, label="latent"):
    """Load golden reference for comparison."""
    golden_file = GOLDEN_DIR / f"step_{step_idx:03d}_{label}.json"
    
    if not golden_file.exists():
        _logger.warning(f"⚡ [split-q][Golden] No reference for step {step_idx} {label}")
        return None
    
    data = json.loads(golden_file.read_text())
    _logger.info(f"⚡ [split-q][Golden] Loaded step {step_idx} {label}: hash={data['latent_hash']}")
    return data


def validate_against_golden(step_idx, latent, label="latent", halt_on_mismatch=True):
    """Compare current latent against golden reference."""
    golden = load_golden_step(step_idx, label)
    
    if golden is None:
        _logger.error(f"⚡ [split-q][Golden] ❌ MISSING REFERENCE for step {step_idx} {label}")
        if halt_on_mismatch:
            raise ValueError(f"Missing golden reference for step {step_idx} {label}")
        return False
    
    current_hash = tensor_hash(latent)
    golden_hash = golden["latent_hash"]
    
    if current_hash == golden_hash:
        _logger.info(f"⚡ [split-q][Golden] ✅ step {step_idx} {label} MATCHES golden")
        return True
    else:
        _logger.error(f"⚡ [split-q][Golden] ❌ step {step_idx} {label} DIVERGED")
        _logger.error(f"⚡ [split-q][Golden]    Golden:  {golden_hash}")
        _logger.error(f"⚡ [split-q][Golden]    Current: {current_hash}")
        
        # Additional diagnostics
        _logger.error(f"⚡ [split-q][Golden]    Shape: golden={golden['latent_shape']} current={list(latent.shape)}")
        
        if halt_on_mismatch:
            raise ValueError(f"Step {step_idx} {label} diverged from golden reference")
        
        return False


def clear_golden_references():
    """Clear all saved golden references."""
    for f in GOLDEN_DIR.glob("step_*.json"):
        f.unlink()
    _logger.info("⚡ [split-q][Golden] Cleared all references")
