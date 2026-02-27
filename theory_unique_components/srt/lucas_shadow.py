"""
Lucas Shadow - The Anti-Phase Operator.

For every constructive phase φ^n, there's a shadow (1-φ)^n.
Lucas numbers L_n = φ^n + (1-φ)^n sum light and shadow.
"""

from srt_library.core.exact import LUCAS_PRIMES, LUCAS_SEQUENCE, PHI_NUMERIC

PHI_CONJUGATE = 1 - PHI_NUMERIC  # ≈ -0.618


def lucas_number(n: int) -> int:
    """Compute L_n = φ^n + (1-φ)^n."""
    if n == 0:
        return 2
    if n == 1:
        return 1

    # Use the sequence definition for efficiency
    if n < len(LUCAS_SEQUENCE):
        return LUCAS_SEQUENCE[n]

    # Extend sequence if needed
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def shadow_phase(n: int) -> float:
    """Compute the shadow phase (1-φ)^n."""
    return PHI_CONJUGATE**n


def is_lucas_prime(n: int) -> bool:
    """Check if L_n is prime."""
    ln = lucas_number(n)
    if ln < 2:
        return False
    if ln == 2 or ln == 3:
        return True
    if ln % 2 == 0 or ln % 3 == 0:
        return False

    i = 5
    while i * i <= ln:
        if ln % i == 0 or ln % (i + 2) == 0:
            return False
        i += 6
    return True


def dark_matter_mass_prediction() -> tuple[float, str]:
    """
    Predict dark matter mass using Lucas boost.

    Uses the ratio L_17/L_13 ≈ 6.85 to boost top quark mass.
    """
    l17 = lucas_number(17)
    l13 = lucas_number(13)
    lucas_boost = l17 / l13
    top_mass = 173.0  # GeV
    prediction = top_mass * lucas_boost / 1000.0  # TeV

    explanation = (
        f"Dark Matter Mass = m_top × (L_17/L_13) = {top_mass} GeV × "
        f"({l17}/{l13}) = {prediction:.2f} TeV"
    )

    return prediction, explanation


def get_shadow_spectrum() -> dict:
    """Get the shadow/dark sector spectrum."""
    return {
        "lucas_primes": LUCAS_PRIMES,
        "dark_matter_candidate": dark_matter_mass_prediction()[0],
        "shadow_phases": [shadow_phase(i) for i in range(10)],
    }
