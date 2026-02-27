"""
Linear algebra operations for Syntonic.

100% NumPy-free and scipy-free implementation.
All operations use native Rust ndarray-linalg backend.

Core operations: eig, svd, qr, solve, inv, expm, logm, etc.
SRT-specific: mm_phi, phi_bracket, mm_corrected, mm_golden_weighted, etc.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from ._core import (
    Structure,
    linalg_bmm,
    linalg_mm,
    linalg_mm_add,
    linalg_mm_corrected,
    linalg_mm_golden_phase,
    linalg_mm_golden_weighted,
    linalg_mm_hn,
    linalg_mm_nh,
    linalg_mm_nt,
    linalg_mm_phi,
    linalg_mm_tn,
    linalg_mm_tt,
    linalg_phi_antibracket,
    linalg_phi_bracket,
    linalg_projection_sum,
)
from .dtype import complex128, float64
from .state import State
from .exceptions import LinAlgError


def matmul(a: State, b: State) -> State:
    """Matrix multiplication."""
    return a @ b


def dot(a: State, b: State) -> float:
    """Dot product of flattened arrays."""
    a_flat = a.to_list()
    b_flat = b.to_list()
    if len(a_flat) != len(b_flat):
        raise LinAlgError("Vectors must have same length for dot product")

    # Handle complex numbers
    if a.dtype.name == "complex128" or b.dtype.name == "complex128":
        result = sum(x * y for x, y in zip(a_flat, b_flat))
        return result
    return float(sum(x * y for x, y in zip(a_flat, b_flat)))


def inner(a: State, b: State) -> Union[float, complex]:
    """Inner product <a|b> (conjugate first argument for complex)."""
    a_flat = a.to_list()
    b_flat = b.to_list()
    if len(a_flat) != len(b_flat):
        raise LinAlgError("Vectors must have same length for inner product")

    if a.dtype.name == "complex128":
        # Conjugate first argument
        return sum(x.conjugate() * y for x, y in zip(a_flat, b_flat))
    return float(sum(x * y for x, y in zip(a_flat, b_flat)))


def outer(a: State, b: State) -> State:
    """Outer product |a><b|."""
    a_flat = a.to_list()
    b_flat = b.to_list()

    # Build outer product matrix
    result = []
    for x in a_flat:
        for y in b_flat:
            result.append(x * y)

    new_shape = (len(a_flat), len(b_flat))
    return State(result, dtype=a.dtype, device=a.device, shape=new_shape)


def norm(x: State, ord: Optional[int] = None) -> float:
    """Vector or matrix norm."""
    return x.norm(ord)


def _state_from_storage(storage, dtype, device, shape):
    """Create State from storage with explicit shape."""
    s = State.__new__(State)
    s._storage = storage
    s._dtype = dtype
    s._device = device
    s._shape = shape
    s._syntony_cache = None
    s._gnosis_cache = None
    return s


def eig(a: State) -> Tuple[State, State]:
    """
    Eigenvalue decomposition.

    Returns:
        (eigenvalues, eigenvectors)
    """
    w_storage, v_storage = a._storage.eig()
    n = a.shape[0]

    # Eigenvalues shape is (n,), eigenvectors shape is (n, n)
    w = _state_from_storage(w_storage, complex128, a.device, (n,))
    v = _state_from_storage(v_storage, complex128, a.device, (n, n))
    return w, v


def eigh(a: State) -> Tuple[State, State]:
    """
    Eigenvalue decomposition for Hermitian matrices.

    Returns:
        (eigenvalues, eigenvectors)
    """
    w_storage, v_storage = a._storage.eigh()
    n = a.shape[0]

    # For Hermitian, eigenvalues are real
    w = _state_from_storage(w_storage, float64, a.device, (n,))
    v = _state_from_storage(v_storage, a.dtype, a.device, (n, n))
    return w, v


def svd(a: State, full_matrices: bool = True) -> Tuple[State, State, State]:
    """
    Singular value decomposition.

    Returns:
        (U, S, Vh)
    """
    u_storage, s_storage, vh_storage = a._storage.svd(full_matrices)
    m, n = a.shape[0], a.shape[1]
    k = min(m, n)

    u = _state_from_storage(
        u_storage, a.dtype, a.device, (m, m) if full_matrices else (m, k)
    )
    s = _state_from_storage(s_storage, float64, a.device, (k,))
    vh = _state_from_storage(
        vh_storage, a.dtype, a.device, (n, n) if full_matrices else (k, n)
    )
    return u, s, vh


def qr(a: State) -> Tuple[State, State]:
    """
    QR decomposition.

    Returns:
        (Q, R) where A = QR
    """
    q_storage, r_storage = a._storage.qr()
    m, n = a.shape[0], a.shape[1]
    k = min(m, n)

    q = _state_from_storage(q_storage, a.dtype, a.device, (m, k))
    r = _state_from_storage(r_storage, a.dtype, a.device, (k, n))
    return q, r


def cholesky(a: State) -> State:
    """
    Cholesky decomposition.

    Returns:
        L where A = LL*
    """
    l_storage = a._storage.cholesky()
    return _state_from_storage(l_storage, a.dtype, a.device, a.shape)


def inv(a: State) -> State:
    """Matrix inverse."""
    inv_storage = a._storage.inv()
    return _state_from_storage(inv_storage, a.dtype, a.device, a.shape)


def pinv(a: State) -> State:
    """
    Moore-Penrose pseudo-inverse.

    Computes via SVD: A^+ = V @ S^+ @ U^H
    """
    # Use full_matrices=True to avoid None values from SVD
    u, s, vh = svd(a, full_matrices=True)

    m, n = a.shape
    k = min(m, n)

    # Invert non-zero singular values
    s_flat = s.to_list()
    tol = max(a.shape) * max(abs(x) for x in s_flat) * 1e-15
    s_inv = [1.0 / x if abs(x) > tol else 0.0 for x in s_flat]

    # Build S^+ matrix (n x m) - transpose of diagonal matrix with inverted singular values
    s_inv_mat = [0.0] * (n * m)
    for i in range(k):
        s_inv_mat[i * m + i] = s_inv[i]

    s_inv_state = State(s_inv_mat, dtype=float64, device=a.device, shape=(n, m))

    # A^+ = Vh^H @ S^+ @ U^H = V @ S^+ @ U^H
    return vh.H @ s_inv_state @ u.H


def det(a: State) -> complex:
    """Matrix determinant."""
    return a._storage.det()


def trace(a: State) -> complex:
    """Matrix trace."""
    return a._storage.trace()


def solve(a: State, b: State) -> State:
    """
    Solve linear system Ax = b.

    Returns:
        x such that Ax = b
    """
    x_storage = a._storage.solve(b._storage)

    # Determine output shape
    if b.ndim == 1:
        out_shape = b.shape
    else:
        out_shape = (a.shape[1], b.shape[1])

    return _state_from_storage(x_storage, a.dtype, a.device, out_shape)


def expm(a: State) -> State:
    """
    Matrix exponential exp(A).

    Important for CRT: evolution operators U(t) = exp(-iHt).

    Uses native Rust ndarray-linalg implementation (no scipy required).
    Returns complex result even for real input.
    """
    result_storage = a._storage.expm()
    # expm always returns complex
    return _state_from_storage(result_storage, complex128, a.device, a.shape)


def logm(a: State) -> State:
    """
    Matrix logarithm log(A).

    Computes the principal matrix logarithm.
    Uses native Rust ndarray-linalg implementation (no scipy required).
    Returns complex result even for real input.
    """
    result_storage = a._storage.logm()
    # logm always returns complex
    return _state_from_storage(result_storage, complex128, a.device, a.shape)


# =============================================================================
# SRT-Specific Matrix Operations (Rust backend)
# =============================================================================


def mm(a: State, b: State) -> State:
    """
    Core matrix multiplication via Rust linalg backend.

    C = A × B
    """
    result = linalg_mm(a._storage, b._storage)
    m, n = a.shape[0], b.shape[1]
    return _state_from_storage(result, a.dtype, a.device, (m, n))


def mm_add(
    a: State, b: State, c: State, alpha: float = 1.0, beta: float = 1.0
) -> State:
    """
    GEMM: General Matrix-Matrix multiplication.

    Result = α × (A × B) + β × C
    """
    result = linalg_mm_add(a._storage, b._storage, c._storage, alpha, beta)
    return _state_from_storage(result, a.dtype, a.device, c.shape)


def mm_tn(a: State, b: State) -> State:
    """Transposed-None matmul: C = Aᵀ × B"""
    result = linalg_mm_tn(a._storage, b._storage)
    m, n = a.shape[1], b.shape[1]
    return _state_from_storage(result, a.dtype, a.device, (m, n))


def mm_nt(a: State, b: State) -> State:
    """None-Transposed matmul: C = A × Bᵀ"""
    result = linalg_mm_nt(a._storage, b._storage)
    m, n = a.shape[0], b.shape[0]
    return _state_from_storage(result, a.dtype, a.device, (m, n))


def mm_tt(a: State, b: State) -> State:
    """Transposed-Transposed matmul: C = Aᵀ × Bᵀ"""
    result = linalg_mm_tt(a._storage, b._storage)
    m, n = a.shape[1], b.shape[0]
    return _state_from_storage(result, a.dtype, a.device, (m, n))


def mm_hn(a: State, b: State) -> State:
    """Hermitian-None matmul: C = A† × B (conjugate transpose of A)"""
    result = linalg_mm_hn(a._storage, b._storage)
    m, n = a.shape[1], b.shape[1]
    return _state_from_storage(result, complex128, a.device, (m, n))


def mm_nh(a: State, b: State) -> State:
    """None-Hermitian matmul: C = A × B†"""
    result = linalg_mm_nh(a._storage, b._storage)
    m, n = a.shape[0], b.shape[0]
    return _state_from_storage(result, complex128, a.device, (m, n))


def bmm(a: State, b: State) -> State:
    """
    Batched matrix multiplication.

    For 3D tensors: C[i] = A[i] × B[i]
    """
    result = linalg_bmm(a._storage, b._storage)
    batch, m, k = a.shape[0], a.shape[1], b.shape[2]
    return _state_from_storage(result, a.dtype, a.device, (batch, m, k))


def mm_phi(a: State, b: State, n: int = 1) -> State:
    """
    φ-scaled matmul: φⁿ × (A × B)

    Uses exact Fibonacci formula for φⁿ via GoldenExact.
    The golden ratio φ = (1 + √5) / 2 is the fundamental recursion eigenvalue.

    Args:
        a: First matrix
        b: Second matrix
        n: Power of φ (default 1)

    Returns:
        φⁿ × (A × B)
    """
    result = linalg_mm_phi(a._storage, b._storage, n)
    m, k = a.shape[0], b.shape[1]
    return _state_from_storage(result, a.dtype, a.device, (m, k))


def phi_bracket(a: State, b: State) -> State:
    """
    Golden commutator (φ-bracket): [A, B]_φ = AB - φ⁻¹BA

    The fundamental bracket for SRT φ-Lie algebra representations.
    Uses exact φ⁻¹ = φ - 1 from GoldenExact.

    This is the core algebraic structure of Syntony Recursion Theory.
    """
    result = linalg_phi_bracket(a._storage, b._storage)
    return _state_from_storage(result, a.dtype, a.device, a.shape)


def phi_antibracket(a: State, b: State) -> State:
    """
    Golden anticommutator: {A, B}_φ = AB + φ⁻¹BA

    Symmetric counterpart to the φ-bracket.
    """
    result = linalg_phi_antibracket(a._storage, b._storage)
    return _state_from_storage(result, a.dtype, a.device, a.shape)


def mm_corrected(a: State, b: State, structure: Structure, sign: int = 1) -> State:
    """
    Correction factor matmul: (1 ± q/N) × (A × B)

    Uses Structure for dimension N. The universal syntony deficit q ≈ 0.027395
    modifies operations based on the algebraic structure dimension.

    Args:
        a: First matrix
        b: Second matrix
        structure: Algebraic structure (e.g., Structure.e8_adjoint() for N=248)
        sign: 1 for (1 + q/N), -1 for (1 - q/N)

    Returns:
        Corrected matrix product

    Example:
        >>> from syntonic_applications.linalg import mm_corrected, Structure
        >>> result = mm_corrected(A, B, Structure.e8_roots(), sign=1)  # (1 + q/240) × AB
    """
    result = linalg_mm_corrected(a._storage, b._storage, structure, sign)
    m, k = a.shape[0], b.shape[1]
    return _state_from_storage(result, a.dtype, a.device, (m, k))


def mm_golden_phase(a: State, b: State, n: int = 1) -> State:
    """
    Complex phase matmul: e^{iπn/φ} × (A × B)

    Applies a golden-ratio-modulated phase rotation.
    Uses π from FundamentalConstant::Pi and φ from GoldenExact.

    Result is complex even for real inputs.
    """
    result = linalg_mm_golden_phase(a._storage, b._storage, n)
    m, k = a.shape[0], b.shape[1]
    return _state_from_storage(result, complex128, a.device, (m, k))


def mm_golden_weighted(a: State, b: State) -> State:
    """
    Golden-weighted matmul: C[i,j] = Σₖ A[i,k] × B[k,j] × exp(−k²/φ)

    Each summation index k is weighted by a golden Gaussian.
    This is the natural inner product for SRT spectral analysis.
    """
    result = linalg_mm_golden_weighted(a._storage, b._storage)
    m, k = a.shape[0], b.shape[1]
    return _state_from_storage(result, a.dtype, a.device, (m, k))


def projection_sum(psi: State, projectors: list, coefficients: list) -> State:
    """
    DHSR projection sum: Ψ + Σₖ αₖ × (Pₖ × Ψ)

    Used for DHSR projection summation over lattice points.
    Each projector Pₖ is applied to the state Ψ, scaled by coefficient αₖ,
    and summed with the original state.

    Args:
        psi: Base state vector/matrix
        projectors: List of projection operators (States)
        coefficients: List of scaling coefficients (floats)

    Returns:
        The projected state
    """
    proj_storages = [p._storage for p in projectors]
    result = linalg_projection_sum(psi._storage, proj_storages, coefficients)
    return _state_from_storage(result, psi.dtype, psi.device, psi.shape)


__all__ = [
    # Basic operations
    "matmul",
    "dot",
    "inner",
    "outer",
    "norm",
    # Decompositions
    "eig",
    "eigh",
    "svd",
    "qr",
    "cholesky",
    # Inverses and solvers
    "inv",
    "pinv",
    "det",
    "trace",
    "solve",
    # Matrix functions
    "expm",
    "logm",
    # SRT-specific operations (Rust backend)
    "mm",
    "mm_add",
    "mm_tn",
    "mm_nt",
    "mm_tt",
    "mm_hn",
    "mm_nh",
    "bmm",
    "mm_phi",
    "phi_bracket",
    "phi_antibracket",
    "mm_corrected",
    "mm_golden_phase",
    "mm_golden_weighted",
    "projection_sum",
    # Structure enum for correction factors
    "Structure",
]
