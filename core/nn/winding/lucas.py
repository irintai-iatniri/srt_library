"""
Lucas Shadow Injection - Implements Pillar V (The Shadow).

Uses Lucas Number ratios to inject 'Shadow Phase' (Dark Energy)
into the network. This prevents static loops (overfitting) and
drives the system toward the attractor.

Identity: L_n = phi^n + (1-phi)^n
"""

from ... import sn
from ..._core import ResonantTensor
from ...exact import PHI


class LucasShadow(sn.Module):
    """
    Applies 'Shadow Pressure' (Novelty/Noise) to the latent state.

    Unlike random noise, Lucas Shadow is geometric anti-phase.
    It simulates the vacuum pressure of the Dark Sector.
    """

    def __init__(self, level: int):
        super().__init__()
        self.level = level
        # Calculate Lucas Shadow magnitude: (1 - phi)^n
        # This is the "Destructive" component of the expansion
        self.shadow_magnitude = abs((1.0 - PHI) ** level)

    def forward(self, x: ResonantTensor) -> ResonantTensor:
        if self.training:
            # Generate Shadow Phase (Anti-aligned geometry)
            # In a real implementation, this would be an orthogonal rotation
            # For now, we model it as structured noise scaled by Lucas gap
            noise = sn.randn_like(x) * self.shadow_magnitude

            # The Shadow exerts pressure on the Light
            return x.elementwise_add(noise)
        return x
