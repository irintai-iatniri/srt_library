import math

import pytest

from syntonic_applications.core.dtype import float64
from syntonic_applications.core.state import State


def test_variance_and_std():
    psi = State([1.0, 3.0, 5.0], dtype=float64)
    expected_variance = 8.0 / 3.0
    assert math.isclose(psi.variance(), expected_variance, rel_tol=1e-9)
    assert math.isclose(psi.std(), math.sqrt(expected_variance), rel_tol=1e-9)


def test_percentile_and_quantile():
    sample = State([1, 2, 3, 4, 5], dtype=float64)
    assert sample.percentile(0.5) == 3.0
    assert sample.quantile(0.25) == 2.0
    assert sample.percentile(1.0) == 5.0
    with pytest.raises(ValueError):
        sample.percentile(1.5)


def test_covariance_and_correlation():
    a = State([1.0, 2.0, 3.0], dtype=float64)
    b = State([2.0, 4.0, 6.0], dtype=float64)
    assert math.isclose(a.covariance(b), 4.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(a.correlation_coefficient(b), 1.0, rel_tol=1e-9)

    with pytest.raises(ValueError):
        a.covariance(State([1, 2]))

    zero_std = State([1.0, 1.0, 1.0], dtype=float64)
    with pytest.raises(ValueError):
        zero_std.correlation_coefficient(b)
