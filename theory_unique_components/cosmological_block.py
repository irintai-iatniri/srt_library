# python/syntonic/nn/architectures/cosmological_block.py


class CosmologicalBlock(sn.Module):
    """
    The Unit of Existence.

    1. Mersenne (Structure): Determines the dimension/capacity.
    2. Fermat (Interaction): Determines the attention/force type.
    3. Lucas (Balance): Inject shadow phase to prevent collapse.
    """

    def __init__(self, generation_idx: int, force_type: int):
        super().__init__()

        # 1. Structure (Mersenne)
        # Uses the MersenneHierarchy to get Prime Dimensions
        self.structure = MersenneLinear(generation_idx)

        # 2. Interaction (Fermat)
        # Gating based on which force this block represents
        self.force_gate = FermatInteractionGate(force_type)

        # 3. Balance (Lucas)
        # Shadow injection based on recursion depth
        self.shadow = LucasShadow(level=generation_idx)

        # The Engine (Phi)
        self.activation = GoldenGELU()

    def forward(self, x):
        x = self.structure(x)  # Matter forms
        x = self.force_gate(x)  # Forces interact
        x = self.activation(x)  # Time evolves
        x = self.shadow(x)  # Vacuum stabilizes
        return x
