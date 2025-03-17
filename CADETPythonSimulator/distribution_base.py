import numpy as np
import numpy.typing as npt
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.exception import CADETPythonSimError

class DistributionBase:
    """Small wrapper class for implementing Distributions for Boundary Conditions."""

    def get_distribution(t: float, section_nr: int)->np.ndarray:
        """Abtract class to call."""
        pass

class ConstantVolumeDistribution(DistributionBase):
    """Implements DistributionBase for a Constant Volume Distribution."""

    def __init__(self, component_system: CPSComponentSystem, c: npt.ArrayLike):
        """
        Construct the Object and calculate the Constant Value.

        Concentration of the Last element is not given.
        """
        if len(c) != component_system.n_comp - 1:
            raise ValueError(
                f"Length of concentrations must be {component_system.n_comp - 1}"
            )

        self.component_system = component_system
        self.c = c

        n = [i for i in self.c]
        m = [
            n_i * self.component_system.molecular_weights[i]
            for i, n_i in enumerate(n)
        ]
        V = [
            m_i / self.component_system.densities[i]
            for i, m_i in enumerate(m)
        ]
        V_solvent = 1 - sum(V)

        if V_solvent<0:
            raise CADETPythonSimError(
                "Last species Volume is negative. Misconfigured Simulation"
                )

        V.append(V_solvent)

        self.V_dist = [
            i / sum(V)
            for i in V
            ]

    def get_distribution(self, t, sec) -> list[float]:
        """Return Constant Volume Distribution."""
        return self.V_dist


class ConstantConcentrationDistribution(DistributionBase):
    """Implements of DistributionBase for a Constant Concentration Distribution."""

    def __init__(self, component_system: CPSComponentSystem, c: npt.ArrayLike):
        """
        Construct the object and calculate the Constant Value.

        Concentration of the Last element is not given.
        """
        if len(c) != component_system.n_comp - 1:
            raise ValueError(
                f"Length of concentrations must be {component_system.n_comp - 1}"
            )

        self.component_system = component_system

        n = [i for i in c]
        m = [
            n_i * component_system.molecular_weights[i]
            for i, n_i in enumerate(n)
        ]
        V = [
            m_i / component_system.densities[i]
            for i, m_i in enumerate(m)
        ]
        V_solvent = 1 - sum(V)

        if V_solvent<0:
            raise CADETPythonSimError(
                "Last species Volume is negative. Misconfigured Simulation"
                )

        m_solvent = component_system.densities[-1] * V_solvent
        n_solvent = m_solvent / component_system.molecular_weights[-1]

        self.c = [*c, n_solvent]


    def get_distribution(self, t, sec) -> list[float]:
        """Return Constant Concentrations."""
        return self.c
