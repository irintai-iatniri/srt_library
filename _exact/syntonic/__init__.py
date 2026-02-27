"""
Syntonic: Tensor Library for Syntony Recursion Theory

The HCB2 computational substrate - where Vibes replace Bits.

States: {1, 0, -1, i} = {Expansion, Vacuum, Contraction, Recursion}
Units: Vibe (1 state) → Chord (4 Vibes) → Octave (8 Vibes / E8 vector)
"""

__version__ = "0.2.0"

# Import the Rust backend
try:
    from syntonic._core import *
    _RUST_AVAILABLE = True
except ImportError as e:
    _RUST_AVAILABLE = False
    _RUST_ERROR = str(e)

def rust_available():
    return _RUST_AVAILABLE
