"""Test prime selection rules."""

from syntonic_applications.srt.prime_selection import (
    FERMAT_PRIMES,
    fermat_number,
    is_fermat_prime,
    is_mersenne_prime,
    lucas_number,
    mersenne_number,
)


def test_fermat_primes():
    assert FERMAT_PRIMES == [3, 5, 17, 257, 65537]
    assert fermat_number(0) == 3
    assert fermat_number(4) == 65537
    assert is_fermat_prime(4)  # F_4 prime
    assert not is_fermat_prime(5)  # F_5 composite


def test_mersenne_primes():
    assert mersenne_number(2) == 3
    assert mersenne_number(7) == 127
    assert is_mersenne_prime(7)
    assert not is_mersenne_prime(11)  # M_11 barrier


def test_lucas_numbers():
    assert lucas_number(0) == 2
    assert lucas_number(1) == 1
    assert lucas_number(5) == 11
