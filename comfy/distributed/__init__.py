"""Distributed runtime for FSDP and sequence parallel inference."""

__version__ = "0.1.0-dev"
__all__ = [
    'MultiprocExecutor',
]

from .executor import MultiprocExecutor
