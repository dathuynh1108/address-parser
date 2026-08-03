"""Public API for the Vietnamese address parser package."""

from .kernels import native_acceleration_available, require_native_acceleration
from .parser import AddressParser

__all__ = [
    "AddressParser",
    "native_acceleration_available",
    "require_native_acceleration",
]
