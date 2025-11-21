"""ProxiedSingleton for nodes module base classes.

Exposes base classes like PreviewImage and SaveImage that custom nodes inherit from.
This is necessary for Crystools, which subclasses these.
"""

import logging
import nodes
from comfy.isolation import LOG_PREFIX

logger = logging.getLogger(__name__)

class NodesProxy:
    """Proxy for nodes base classes (Crystools subset).
    
    This is NOT a ProxiedSingleton yet - it's a simple wrapper for testing.
    It just forwards class definitions, no RPC needed.
    """
    
    # Expose base classes directly
    PreviewImage = nodes.PreviewImage
    SaveImage = nodes.SaveImage
    
    def __init__(self):
        super().__init__()
        logger.debug(f"{LOG_PREFIX}[NodesProxy] Initialized with PreviewImage, SaveImage")

