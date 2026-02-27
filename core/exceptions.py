"""Custom exceptions for Syntonic."""

from __future__ import annotations


class SyntonicError(Exception):
    """Base exception for Syntonic library."""

    pass


class DeviceError(SyntonicError):
    """Exception related to device operations."""

    pass


class DTypeError(SyntonicError):
    """Exception related to data type operations."""

    pass


class ShapeError(SyntonicError):
    """Exception related to tensor shape operations."""

    pass


class LinAlgError(SyntonicError):
    """Exception related to linear algebra operations."""

    pass
