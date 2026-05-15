from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Species:
    """
    A specific chemical form of a component.

    Parameters
    ----------
    name : str
    charge : int
    molar_mass : float, optional
        Molar mass [kg/mol].
    density : float, optional
        Pure-component density [kg/m³].
    viscosity : float, optional
        Pure-component dynamic viscosity [Pa·s].
    c_ref : float, optional
        Standard-state concentration [mol/m³].
        Defaults to ``density / molar_mass`` when both are set, else 1000 mol/m³.
    """

    name: str
    charge: int = 0
    molar_mass: Optional[float] = None
    density: Optional[float] = None
    viscosity: Optional[float] = None
    c_ref: Optional[float] = None

    def __post_init__(self) -> None:
        if self.c_ref is None:
            if self.density is not None and self.molar_mass is not None:
                self.c_ref = self.density / self.molar_mass
            else:
                self.c_ref = 1000.0


class Component:
    """
    A conserved chemical entity that may exist as multiple species.

    Parameters
    ----------
    name : str
    species : list[Species], optional
        If omitted, a single Species named ``name`` is created from ``**kwargs``.

    Examples
    --------
    Component("water", molar_mass=0.018015, density=1000.0, viscosity=0.001)

    Component("phosphate", [
        Species("H2PO4-", charge=-1, molar_mass=0.095),
        Species("HPO4-2", charge=-2, molar_mass=0.094),
    ])
    """

    def __init__(
        self, name: str, species: list[Species] | None = None, **kwargs
    ) -> None:
        self.name = name
        self.species = species if species is not None else [Species(name, **kwargs)]
        if not self.species:
            raise ValueError(f"Component '{name}' needs at least one species.")
        names = [s.name for s in self.species]
        if len(names) != len(set(names)):
            raise ValueError(f"Component '{name}' has duplicate species names.")

    def _single_attr(self, attr: str):
        if len(self.species) != 1:
            raise AttributeError(
                f"Component '{self.name}' has {len(self.species)} species; "
                f"use .species[i].{attr} explicitly."
            )
        return getattr(self.species[0], attr)

    @property
    def charge(self) -> int:
        return self._single_attr("charge")

    @property
    def molar_mass(self) -> Optional[float]:
        return self._single_attr("molar_mass")

    @property
    def density(self) -> Optional[float]:
        return self._single_attr("density")

    @property
    def viscosity(self) -> Optional[float]:
        return self._single_attr("viscosity")

    @property
    def c_ref(self) -> float:
        return self._single_attr("c_ref")

    def __repr__(self) -> str:
        return f"Component('{self.name}', species={[s.name for s in self.species]})"


class ComponentSystem:
    """
    An ordered collection of components defining the chemical system.

    Parameters
    ----------
    components : list[Component]
    name : str, optional

    Properties
    ----------
    n_comp : int
        Total number of species (sum over all components).
    names : list[str]
        Species names in order.
    molar_masses : list[float | None]
        Molar masses [kg/mol] per species.
    densities : list[float | None]
        Pure-component densities [kg/m³] per species.
    viscosities : list[float | None]
        Pure-component viscosities [Pa·s] per species.
    charges : list[int]
        Charges per species.
    """

    def __init__(self, components: list[Component], name: str = "") -> None:
        self.name = name
        self._components = list(components)
        names = [s.name for s in self.species]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate species names in ComponentSystem.")

    @property
    def components(self) -> list[Component]:
        return self._components

    @property
    def species(self) -> list[Species]:
        return [s for comp in self._components for s in comp.species]

    @property
    def n_comp(self) -> int:
        return len(self.species)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.species]

    def _collect(self, attr: str) -> list:
        return [getattr(s, attr) for s in self.species]

    @property
    def molar_masses(self) -> list[Optional[float]]:
        return self._collect("molar_mass")

    @property
    def densities(self) -> list[Optional[float]]:
        return self._collect("density")

    @property
    def viscosities(self) -> list[float]:
        return [
            s.viscosity if s.viscosity is not None else np.nan for s in self.species
        ]

    @property
    def charges(self) -> list[int]:
        return self._collect("charge")

    def __repr__(self) -> str:
        return f"ComponentSystem({[c.name for c in self._components]!r})"
