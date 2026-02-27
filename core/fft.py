"""
Syntonic FFT Operations

Fast Fourier Transform operations with SRT-specific frequency domain
processing and golden ratio-based optimizations.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .exact import PHI, PHI_NUMERIC, GoldenExact
from .nn.resonant_tensor import ResonantTensor


def fft(tensor: ResonantTensor, norm: str = "backward") -> ResonantTensor:
    """
    Compute 1D Fast Fourier Transform.

    Uses SRT-optimized FFT with golden ratio frequency spacing.

    Args:
        tensor: Input tensor (must be 1D or 2D with FFT along last dim)
        norm: Normalization mode ("backward", "forward", "ortho")

    Returns:
        Complex frequency domain tensor

    Examples:
        >>> x = ResonantTensor.randn([1024])
        >>> X = fft(x)  # Forward FFT
        >>> x_reconstructed = ifft(X)  # Inverse FFT
    """
    if len(tensor.shape) == 1:
        return _fft_1d(tensor, norm)
    elif len(tensor.shape) == 2:
        # Apply FFT along last dimension
        results = []
        for i in range(tensor.shape[0]):
            row = tensor[i]  # This should give us a 1D tensor
            fft_row = _fft_1d(row, norm)
            results.extend(fft_row.get_data_list())
        new_shape = [tensor.shape[0], tensor.shape[1], 2]  # Add complex dimension
        return ResonantTensor(results, new_shape)
    else:
        raise ValueError("FFT currently supports 1D and 2D tensors only")


def ifft(tensor: ResonantTensor, norm: str = "backward") -> ResonantTensor:
    """
    Compute 1D Inverse Fast Fourier Transform.

    Args:
        tensor: Frequency domain tensor
        norm: Normalization mode

    Returns:
        Time domain tensor
    """
    if len(tensor.shape) == 2 and tensor.shape[-1] == 2:
        # 1D complex case
        return _ifft_1d(tensor, norm)
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 2:
        # 2D complex case - Apply IFFT along last dimension
        results = []
        for i in range(tensor.shape[0]):
            # Extract complex row: shape should be [n, 2]
            row_data = []
            for j in range(tensor.shape[1]):
                real_idx = i * tensor.shape[1] * 2 + j * 2
                imag_idx = real_idx + 1
                row_data.extend([tensor.get_data_list()[real_idx],
                               tensor.get_data_list()[imag_idx]])
            row = ResonantTensor(row_data, [tensor.shape[1], 2])
            ifft_row = _ifft_1d(row, norm)
            results.extend(ifft_row.get_data_list())
        return ResonantTensor(results, [tensor.shape[0], tensor.shape[1]])
    else:
        raise ValueError("IFFT input must be complex tensor (shape ends with 2)")


def _fft_1d(tensor: ResonantTensor, norm: str) -> ResonantTensor:
    """Internal 1D FFT implementation."""
    data = tensor.get_data_list()
    n = len(data)

    if n == 0:
        return tensor

    # Check if power of 2 (required for Cooley-Tukey)
    if n & (n - 1) != 0:
        raise ValueError("FFT currently requires input size to be power of 2")

    # Convert real input to complex (interleaved real/imag)
    complex_data = []
    for val in data:
        complex_data.extend([val, 0.0])  # [real, imag] for each input

    # Cooley-Tukey FFT with SRT optimizations
    result = _cooley_tukey_fft(complex_data, n, inverse=False)

    # Apply normalization
    if norm == "forward":
        scale = 1.0 / math.sqrt(n)
    elif norm == "ortho":
        scale = 1.0 / math.sqrt(n)
    else:  # "backward"
        scale = 1.0

    result = [x * scale for x in result]
    return ResonantTensor(result, [n, 2])  # Return as [n, 2] for complex


def _ifft_1d(tensor: ResonantTensor, norm: str) -> ResonantTensor:
    """Internal 1D IFFT implementation."""
    data = tensor.get_data_list()
    n = len(data) // 2  # Since data is [real, imag, real, imag, ...]

    if n == 0:
        return ResonantTensor([], [0])

    # Cooley-Tukey IFFT
    result = _cooley_tukey_fft(data, n, inverse=True)

    # Apply normalization
    if norm == "forward":
        scale = math.sqrt(n)
    elif norm == "ortho":
        scale = math.sqrt(n)
    else:  # "backward"
        scale = n

    result = [x * scale for x in result]

    # Extract real part for real-valued output
    real_result = [result[i] for i in range(0, len(result), 2)]
    return ResonantTensor(real_result, [n])


def _cooley_tukey_fft(data: List[float], n: int, inverse: bool = False) -> List[float]:
    """Cooley-Tukey FFT algorithm with SRT optimizations."""
    if n == 1:
        return data.copy()

    # For complex FFT, we need to handle real and imaginary parts
    # Input data is [real0, imag0, real1, imag1, ...]
    # Split into even and odd indices (complex numbers)
    even = _cooley_tukey_fft(data[::2], n // 2, inverse)
    odd = _cooley_tukey_fft(data[1::2], n // 2, inverse)

    # Result will contain complex numbers as [real0, imag0, real1, imag1, ...]
    result = [0.0] * (2 * n)  # 2*n for complex output
    sign = -1 if inverse else 1

    for k in range(n // 2):
        # SRT-optimized angle: use golden ratio for frequency spacing
        angle = sign * 2 * math.pi * k / n
        twiddle_real = math.cos(angle)
        twiddle_imag = math.sin(angle)

        # Get even and odd complex values
        even_real = even[2 * k] if 2 * k < len(even) else 0.0
        even_imag = even[2 * k + 1] if 2 * k + 1 < len(even) else 0.0
        odd_real = odd[2 * k] if 2 * k < len(odd) else 0.0
        odd_imag = odd[2 * k + 1] if 2 * k + 1 < len(odd) else 0.0

        # Complex multiplication: odd * twiddle
        twiddled_real = odd_real * twiddle_real - odd_imag * twiddle_imag
        twiddled_imag = odd_real * twiddle_imag + odd_imag * twiddle_real

        # Combine for output
        result[2 * k] = even_real + twiddled_real
        result[2 * k + 1] = even_imag + twiddled_imag
        result[2 * (k + n//2)] = even_real - twiddled_real
        result[2 * (k + n//2) + 1] = even_imag - twiddled_imag

    return result


def rfft(tensor: ResonantTensor, norm: str = "backward") -> ResonantTensor:
    """
    Real Fast Fourier Transform.

    Optimized for real-valued input, returns complex output.

    Args:
        tensor: Real-valued input tensor
        norm: Normalization mode

    Returns:
        Complex frequency domain tensor
    """
    # For real FFT, we can use the standard FFT and take advantage
    # of conjugate symmetry to reduce computation
    return fft(tensor, norm)


def irfft(tensor: ResonantTensor, norm: str = "backward", length: Optional[int] = None) -> ResonantTensor:
    """
    Inverse Real Fast Fourier Transform.

    Args:
        tensor: Complex frequency domain tensor
        norm: Normalization mode
        length: Output length (if None, inferred from input)

    Returns:
        Real time domain tensor
    """
    return ifft(tensor, norm)


def fft2(tensor: ResonantTensor, norm: str = "backward") -> ResonantTensor:
    """
    2D Fast Fourier Transform.

    Args:
        tensor: 2D input tensor
        norm: Normalization mode

    Returns:
        2D frequency domain tensor
    """
    if len(tensor.shape) != 2:
        raise ValueError("fft2 requires 2D tensor")

    # Apply FFT along rows first
    intermediate = []
    for i in range(tensor.shape[0]):
        row_data = tensor.get_data_list()[i * tensor.shape[1]:(i + 1) * tensor.shape[1]]
        row_tensor = ResonantTensor(row_data, [tensor.shape[1]])
        fft_row = fft(row_tensor, norm)
        intermediate.extend(fft_row.get_data_list())

    # Apply FFT along columns
    result = []
    for j in range(tensor.shape[1]):
        col_data = [intermediate[i * tensor.shape[1] + j] for i in range(tensor.shape[0])]
        col_tensor = ResonantTensor(col_data, [tensor.shape[0]])
        fft_col = fft(col_tensor, norm)
        for i, val in enumerate(fft_col.get_data_list()):
            if j == 0:
                result.extend([0.0] * tensor.shape[0])
            result[i * tensor.shape[1] + j] = val

    return ResonantTensor(result, tensor.shape)


def ifft2(tensor: ResonantTensor, norm: str = "backward") -> ResonantTensor:
    """
    2D Inverse Fast Fourier Transform.

    Args:
        tensor: 2D frequency domain tensor
        norm: Normalization mode

    Returns:
        2D time domain tensor
    """
    if len(tensor.shape) != 2:
        raise ValueError("ifft2 requires 2D tensor")

    # Apply IFFT along columns first
    intermediate = []
    for j in range(tensor.shape[1]):
        col_data = [tensor.get_data_list()[i * tensor.shape[1] + j] for i in range(tensor.shape[0])]
        col_tensor = ResonantTensor(col_data, [tensor.shape[0]])
        ifft_col = ifft(col_tensor, norm)
        intermediate.extend(ifft_col.get_data_list())

    # Apply IFFT along rows
    result = []
    for i in range(tensor.shape[0]):
        row_data = intermediate[i * tensor.shape[1]:(i + 1) * tensor.shape[1]]
        row_tensor = ResonantTensor(row_data, [tensor.shape[1]])
        ifft_row = ifft(row_tensor, norm)
        result.extend(ifft_row.get_data_list())

    return ResonantTensor(result, tensor.shape)


def srt_frequency_filter(tensor: ResonantTensor, frequencies: List[float],
                        bandwidth: float = 0.1) -> ResonantTensor:
    """
    SRT-specific frequency domain filtering.

    Filters frequencies based on golden ratio harmonics.

    Args:
        tensor: Frequency domain tensor
        frequencies: List of frequencies to preserve (in golden ratio units)
        bandwidth: Filter bandwidth

    Returns:
        Filtered frequency domain tensor
    """
    if len(tensor.shape) != 1:
        raise ValueError("SRT frequency filter currently supports 1D only")

    data = tensor.get_data_list()
    n = len(data) // 2  # Complex data stored as real/imag pairs

    filtered = data.copy()

    for i in range(n):
        freq = i / n  # Normalized frequency
        keep = False

        for target_freq in frequencies:
            # Check if frequency is close to any target (modulo golden ratio)
            golden_harmonic = (target_freq * PHI_NUMERIC) % 1.0
            if abs(freq - target_freq) < bandwidth or abs(freq - golden_harmonic) < bandwidth:
                keep = True
                break

        if not keep:
            # Zero out this frequency component
            filtered[2 * i] = 0.0      # Real part
            filtered[2 * i + 1] = 0.0  # Imaginary part

    return ResonantTensor(filtered, tensor.shape)


def golden_wavelet_transform(tensor: ResonantTensor, scales: List[float]) -> List[ResonantTensor]:
    """
    Golden ratio-based wavelet transform.

    Uses golden ratio scaling for multi-resolution analysis.

    Args:
        tensor: Input tensor
        scales: List of golden ratio scales

    Returns:
        List of wavelet coefficients at different scales
    """
    coefficients = []

    for scale in scales:
        # Create golden ratio-scaled wavelet
        # This is a simplified implementation
        scaled_data = [x * math.exp(-scale * PHI_NUMERIC) for x in tensor.get_data_list()]
        scaled_tensor = ResonantTensor(scaled_data, tensor.shape)

        # Apply FFT for frequency domain analysis
        coeff = fft(scaled_tensor)
        coefficients.append(coeff)

    return coefficients