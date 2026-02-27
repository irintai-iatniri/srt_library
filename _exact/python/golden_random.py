"""
Golden Ratio Random Number Generator

Integer-only random number generation using golden ratio sequences.
Replaces standard library random with SRT-compatible implementation.
"""

# Fibonacci constants for golden ratio approximation
PHI_NUM = 1597  # F(17)
PHI_DEN = 987   # F(16)


class GoldenRandom:
    """Integer-only golden ratio PRNG."""
    
    def __init__(self, seed: int = 1597):
        self._state = seed & 0xFFFFFFFF
    
    def seed(self, s: int):
        """Set the random seed."""
        self._state = s & 0xFFFFFFFF
    
    def _next(self) -> int:
        """Generate next random integer using golden ratio LCG."""
        # Linear congruential with Fibonacci multiplier
        self._state = (self._state * PHI_NUM + PHI_DEN) & 0xFFFFFFFF
        return self._state
    
    def random(self) -> int:
        """Return random int 0-1023 (scaled by 1024)."""
        return self._next() & 1023
    
    def uniform(self, lo: int = 0, hi: int = 1024) -> int:
        """Return random int in [lo, hi)."""
        return lo + (self._next() % (hi - lo))
    
    def randint(self, lo: int, hi: int) -> int:
        """Return random int in [lo, hi] inclusive."""
        return lo + (self._next() % (hi - lo + 1))
    
    def choice(self, seq):
        """Return random element from non-empty sequence."""
        if not seq:
            raise IndexError("Cannot choose from empty sequence")
        return seq[self._next() % len(seq)]
    
    def shuffle(self, seq):
        """Shuffle sequence in place using Fisher-Yates."""
        for i in range(len(seq) - 1, 0, -1):
            j = self._next() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq
    
    def sample(self, population, k: int):
        """Return k unique random elements from population."""
        n = len(population)
        if k > n:
            raise ValueError("Sample larger than population")
        result = []
        selected = set()
        while len(result) < k:
            idx = self._next() % n
            if idx not in selected:
                selected.add(idx)
                result.append(population[idx])
        return result
    
    def gauss(self, mu: int = 0, sigma: int = 1024) -> int:
        """
        Gaussian approximation using sum of uniform values.
        Returns integer centered at mu with spread sigma.
        """
        # Box-Muller alternative: sum of 12 uniforms approximates normal
        total = sum(self._next() & 0xFFF for _ in range(12))
        # Normalize: mean of 12 * 2048 = 24576, stddev ~ 1024
        normalized = total - 24576
        return mu + (normalized * sigma) // 1024


# Global instance
_rng = GoldenRandom()

# Module-level functions matching random API
def seed(s: int):
    _rng.seed(s)

def random() -> int:
    return _rng.random()

def uniform(lo: int = 0, hi: int = 1024) -> int:
    return _rng.uniform(lo, hi)

def randint(lo: int, hi: int) -> int:
    return _rng.randint(lo, hi)

def choice(seq):
    return _rng.choice(seq)

def shuffle(seq):
    return _rng.shuffle(seq)

def sample(population, k: int):
    return _rng.sample(population, k)

def gauss(mu: int = 0, sigma: int = 1024) -> int:
    return _rng.gauss(mu, sigma)