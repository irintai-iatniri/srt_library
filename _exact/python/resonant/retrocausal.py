"""
Retrocausal Attractor-Guided RES

Theory-pure alternative to backpropagation that uses geometric temporal influence
instead of gradient flow. High-syntony states discovered during evolution are stored
as "attractors" that exert retrocausal influence on harmonization.

From CRT_Altruxa_Bridge.md §17: Future high-syntony states reach backward through
the DHSR cycle to guide parameter evolution.

# Mathematical Formulation

Standard harmonization:
    Ĥ[ψ]ₙ = ψₙ × (1 - β(S) × (1 - w(n)))

Retrocausal harmonization:
    Ĥ_retro[ψ]ₙ = (1 - λ_retro) × Ĥ[ψ]ₙ + λ_retro × Σᵢ wᵢ × (Aᵢ,ₙ - ψₙ)

Where:
- Aᵢ = attractor i lattice values
- wᵢ = weight of attractor i (syntony² × temporal_decay)
- λ_retro = retrocausal pull strength

Expected performance: 15-30% faster convergence compared to standard RES.

# Example Usage

```python
from syntonic_applications.resonant import ResonantTensor
from syntonic_applications.resonant.retrocausal import create_retrocausal_evolver, RetrocausalConfig

# Create tensor
tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])

# Create retrocausal evolver with recommended defaults
evolver = create_retrocausal_evolver(
    tensor,
    population_size=32,
    attractor_capacity=32,
    pull_strength=0.3
)

# Run evolution
result = evolver.run()
print(f"Final syntony: {result.final_syntony:.4f}")
print(f"Converged in {result.generations} generations")
```

# Parameter Tuning Guide

**attractor_capacity**: Number of attractors to retain
- Small (16-32): Fast, memory-efficient, less diverse
- Medium (32-64): Balanced (recommended)
- Large (64-128): More diverse, slower lookups

**pull_strength** (λ_retro): How strongly attractors influence harmonization
- Low (0.1-0.2): Gentle guidance, slower convergence
- Medium (0.3-0.4): Balanced (recommended)
- High (0.5-0.7): Strong pull, risk of premature convergence

**attractor_min_syntony**: Minimum syntony threshold for attractor storage
- Low (0.5-0.6): Many attractors, more noise
- Medium (0.7-0.8): Balanced (recommended)
- High (0.8-0.9): Few attractors, high quality

**attractor_decay_rate**: How quickly old attractors fade
- Low (0.90-0.95): Rapid forgetting, adaptive
- Medium (0.96-0.98): Balanced (recommended)
- High (0.99-1.0): Long memory, less adaptive
"""

from typing import Optional

from applications._core import RESConfig, ResonantEvolver, ResonantTensor, SymExpr, GoldenExact, Rational


class RetrocausalConfig:
    """
    Convenience configuration for retrocausal RES.

    Wraps RESConfig with retrocausal-specific defaults.
    """

    def __init__(
        self,
        # Standard RES parameters
        population_size: int = 32,
        survivor_count: Optional[int] = None,
        lambda_val: Optional[float] = None,
        mutation_scale: float = Rational(1, 10),
        noise_scale: float = Rational(1, 100),
        precision: int = 100,
        max_generations: int = 1000,
        convergence_threshold: float = 1e-6,  # eval boundary
        # Retrocausal parameters
        attractor_capacity: int = 32,
        attractor_pull_strength: float = Rational(3, 10),
        attractor_min_syntony: float = Rational(7, 10),
        attractor_decay_rate: float = Rational(49, 50),  # 0.98 = 49/50
    ):
        """
        Create retrocausal RES configuration.

        Args:
            population_size: Number of mutants per generation (default: 32)
            survivor_count: Number of survivors after filtering (default: population_size // 4)
            lambda_val: Universal syntony deficit (default: Q_DEFICIT ~ 0.027395)
            mutation_scale: Standard deviation of mutations (default: 0.1)
            noise_scale: Noise amplitude for D-phase (default: 0.01)
            precision: Precision for crystallization (default: 100)
            max_generations: Maximum evolution steps (default: 1000)
            convergence_threshold: Syntony change threshold for convergence (default: 1e-6)

            attractor_capacity: Maximum number of attractors to retain (default: 32)
            attractor_pull_strength: Strength of attractor influence, λ_retro ∈ [0,1] (default: 0.3)
            attractor_min_syntony: Minimum syntony for attractor storage (default: 0.7)
            attractor_decay_rate: Temporal decay per generation (default: 0.98)
        """
        self.population_size = population_size
        self.survivor_count = survivor_count or (population_size // 4)
        self.lambda_val = lambda_val
        self.mutation_scale = mutation_scale
        self.noise_scale = noise_scale
        self.precision = precision
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold

        self.attractor_capacity = attractor_capacity
        self.attractor_pull_strength = attractor_pull_strength
        self.attractor_min_syntony = attractor_min_syntony
        self.attractor_decay_rate = attractor_decay_rate

    def to_res_config(self) -> RESConfig:
        """Convert to RESConfig with retrocausal enabled."""
        return RESConfig(
            population_size=self.population_size,
            survivor_count=self.survivor_count,
            lambda_val=self.lambda_val,
            mutation_scale=self.mutation_scale,
            noise_scale=self.noise_scale,
            precision=self.precision,
            max_generations=self.max_generations,
            convergence_threshold=self.convergence_threshold,
            enable_retrocausal=True,
            attractor_capacity=self.attractor_capacity,
            attractor_pull_strength=self.attractor_pull_strength,
            attractor_min_syntony=self.attractor_min_syntony,
            attractor_decay_rate=self.attractor_decay_rate,
        )


def create_retrocausal_evolver(
    template: ResonantTensor,
    population_size: int = 32,
    attractor_capacity: int = 32,
    pull_strength: float = Rational(3, 10),
    min_syntony: float = Rational(7, 10),
    decay_rate: float = Rational(49, 50),
    **kwargs,
) -> ResonantEvolver:
    """
    Create a ResonantEvolver with retrocausal attractor guidance.

    This is the recommended way to create retrocausal evolvers with sensible defaults.

    Args:
        template: Template tensor defining shape and mode structure
        population_size: Number of mutants per generation (default: 32)
        attractor_capacity: Maximum number of attractors (default: 32)
        pull_strength: Attractor influence strength, λ_retro ∈ [0,1] (default: 0.3)
        min_syntony: Minimum syntony for attractor storage (default: 0.7)
        decay_rate: Temporal decay rate (default: 0.98)
        **kwargs: Additional RESConfig parameters

    Returns:
        ResonantEvolver configured for retrocausal operation

    Example:
        >>> from syntonic_applications.resonant import ResonantTensor
        >>> from syntonic_applications.resonant.retrocausal import create_retrocausal_evolver
        >>>
        >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        >>> evolver = create_retrocausal_evolver(tensor)
        >>> result = evolver.run()
    """
    config = RESConfig(
        population_size=population_size,
        enable_retrocausal=True,
        attractor_capacity=attractor_capacity,
        attractor_pull_strength=pull_strength,
        attractor_min_syntony=min_syntony,
        attractor_decay_rate=decay_rate,
        **kwargs,
    )

    # Unwrap Python wrapper if present
    if hasattr(template, "_inner"):
        template = template._inner

    return ResonantEvolver(template, config)


def create_standard_evolver(
    template: ResonantTensor, population_size: int = 32, **kwargs
) -> ResonantEvolver:
    """
    Create a standard ResonantEvolver WITHOUT retrocausal guidance.

    Useful for comparison benchmarks.

    Args:
        template: Template tensor defining shape and mode structure
        population_size: Number of mutants per generation (default: 32)
        **kwargs: Additional RESConfig parameters

    Returns:
        ResonantEvolver configured for standard RES (no retrocausal)

    Example:
        >>> from syntonic_applications.resonant import ResonantTensor
        >>> from syntonic_applications.resonant.retrocausal import create_standard_evolver
        >>>
        >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        >>> evolver = create_standard_evolver(tensor)
        >>> result = evolver.run()
    """
    config = RESConfig(
        population_size=population_size, enable_retrocausal=False, **kwargs
    )

    # Unwrap Python wrapper if present
    if hasattr(template, "_inner"):
        template = template._inner

    return ResonantEvolver(template, config)


def compare_convergence(
    template: ResonantTensor,
    population_size: int = 32,
    max_generations: int = 500,
    trials: int = 5,
) -> dict:
    """
    Compare convergence speed between standard and retrocausal RES.

    Runs multiple trials of both approaches and reports statistics.

    Args:
        template: Template tensor for evolution
        population_size: Population size for both approaches
        max_generations: Maximum generations per trial
        trials: Number of trials per approach

    Returns:
        Dictionary with convergence statistics:
        - 'standard': {'mean_gens': float, 'mean_syntony': float, 'results': [RESResult]}
        - 'retrocausal': {'mean_gens': float, 'mean_syntony': float, 'results': [RESResult]}
        - 'speedup': float (relative improvement)

    Example:
        >>> from syntonic_applications.resonant import ResonantTensor
        >>> from syntonic_applications.resonant.retrocausal import compare_convergence
        >>>
        >>> tensor = ResonantTensor([1.0, 2.0, 3.0, 4.0], [4])
        >>> stats = compare_convergence(tensor, trials=10)
        >>> print(f"Speedup: {stats['speedup']:.1%}")
    """
    standard_results = []
    retrocausal_results = []

    # Run standard RES trials
    for _ in range(trials):
        evolver = create_standard_evolver(
            template, population_size=population_size, max_generations=max_generations
        )
        result = evolver.run()
        standard_results.append(result)

    # Run retrocausal RES trials
    for _ in range(trials):
        evolver = create_retrocausal_evolver(
            template, population_size=population_size, max_generations=max_generations
        )
        result = evolver.run()
        retrocausal_results.append(result)

    # Compute statistics
    standard_mean_gens = sum(r.generations for r in standard_results) / trials
    standard_mean_syntony = sum(r.final_syntony for r in standard_results) / trials

    retrocausal_mean_gens = sum(r.generations for r in retrocausal_results) / trials
    retrocausal_mean_syntony = (
        sum(r.final_syntony for r in retrocausal_results) / trials
    )

    speedup = (standard_mean_gens - retrocausal_mean_gens) / standard_mean_gens

    return {
        "standard": {
            "mean_generations": standard_mean_gens,
            "mean_syntony": standard_mean_syntony,
            "results": standard_results,
        },
        "retrocausal": {
            "mean_generations": retrocausal_mean_gens,
            "mean_syntony": retrocausal_mean_syntony,
            "results": retrocausal_results,
        },
        "speedup": speedup,
    }


__all__ = [
    "RetrocausalConfig",
    "create_retrocausal_evolver",
    "create_standard_evolver",
    "compare_convergence",
]
