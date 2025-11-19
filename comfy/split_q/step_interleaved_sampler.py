"""Step-interleaved sampler that alternates models per denoising step."""

import logging
import torch

_logger = logging.getLogger(__name__)


class StepInterleavedWrapper:
    """Wraps two CFGGuiders to alternate per denoising step."""
    
    def __init__(self, guider_0, guider_1):
        self.guider_0 = guider_0
        self.guider_1 = guider_1
        self.step_count = 0
        
    def __call__(self, x, sigma, **kwargs):
        """Route to alternating model based on step count."""
        active_model = self.step_count % 2
        self.step_count += 1
        
        if active_model == 0:
            _logger.info("⚡ [split-q][StepInterleaved] step %d -> model_0", self.step_count - 1)
            return self.guider_0(x, sigma, **kwargs)
        else:
            _logger.info("⚡ [split-q][StepInterleaved] step %d -> model_1", self.step_count - 1)
            # Transfer to cuda:1
            x_1 = x.to(self.guider_1.model_patcher.load_device)
            result = self.guider_1(x_1, sigma, **kwargs)
            # Transfer back to cuda:0
            return result.to(self.guider_0.model_patcher.load_device)
    
    def __getattr__(self, name):
        """Delegate other attributes to guider_0."""
        return getattr(self.guider_0, name)
