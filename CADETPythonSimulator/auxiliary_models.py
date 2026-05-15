import numpy as np


class IonicStrengthModel:
    """Stub demonstrating model-owned auxiliary state declaration.

    A real implementation would compute I = 0.5 * sum(c_i * z_i^2) using
    per-component charges from ComponentSystem.charges.
    """

    aux_structures = {"ionic_strength": {"entries": {"I": 1}}}

    def compute(self, c: np.ndarray) -> np.ndarray:
        """Return ionic strength. Stub returns zeros."""
        return np.zeros(1)
