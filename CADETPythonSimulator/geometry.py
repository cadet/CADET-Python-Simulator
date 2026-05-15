from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# ── Dimension boundary types ───────────────────────────────────────────────


@dataclass(frozen=True)
class DimBoundary:
    """Describes what lives at the low and high face of a dimension."""

    low: str  # "inlet", "outlet", "symmetry", "wall", "flux_interface"
    high: str


# ── Physical geometries ────────────────────────────────────────────────────


class PhysicalGeometry(ABC):
    """Physical shape of a domain. Defines dimensions, extents, and boundaries."""

    natural_dims: list[str]

    @abstractmethod
    def extent(self, dim: str) -> tuple[float, float]:
        """Physical extent (low, high) of a dimension."""

    @abstractmethod
    def boundary(self, dim: str) -> DimBoundary:
        """Boundary type at each face of a dimension."""

    def coordinates(self, dim: str, n_cells: int) -> np.ndarray:
        lo, hi = self.extent(dim)
        return np.linspace(lo, hi, n_cells)

    def spatially_varying(
        self, n_cells: dict[str, int] | None = None
    ) -> dict[str, np.ndarray]:
        """Spatially varying physical quantities (e.g. cross-section for Frustum)."""
        return {}


class Cylinder(PhysicalGeometry):
    """Straight cylinder. Flow along axial direction."""

    natural_dims = ["axial", "radial", "angular"]

    def __init__(self, length: float, radius: float = None):
        self.length = length
        self.radius = radius

    def extent(self, dim: str) -> tuple[float, float]:
        if dim == "axial":
            return (0.0, self.length)
        if dim == "radial":
            return (0.0, self.radius)
        if dim == "angular":
            return (0.0, 2 * np.pi)
        raise ValueError(f"Unknown dim: {dim}")

    def boundary(self, dim: str) -> DimBoundary:
        if dim == "axial":
            return DimBoundary("inlet", "outlet")
        if dim == "radial":
            return DimBoundary("symmetry", "wall")
        if dim == "angular":
            return DimBoundary("periodic", "periodic")
        raise ValueError(f"Unknown dim: {dim}")


class HollowCylinder(PhysicalGeometry):
    """Hollow cylinder (annulus). Flow from outer to inner radius."""

    natural_dims = ["radial", "axial", "angular"]

    def __init__(self, r_inner: float, r_outer: float, length: float = None):
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.length = length

    def extent(self, dim: str) -> tuple[float, float]:
        if dim == "radial":
            return (self.r_inner, self.r_outer)
        if dim == "axial":
            return (0.0, self.length)
        if dim == "angular":
            return (0.0, 2 * np.pi)
        raise ValueError(f"Unknown dim: {dim}")

    def boundary(self, dim: str) -> DimBoundary:
        if dim == "radial":
            return DimBoundary("outlet", "inlet")  # inner=outlet, outer=inlet
        if dim == "axial":
            return DimBoundary("wall", "wall")
        if dim == "angular":
            return DimBoundary("periodic", "periodic")
        raise ValueError(f"Unknown dim: {dim}")


class Frustum(PhysicalGeometry):
    """Cylinder with linearly varying radius (tapered column).

    Velocity u(z) = Q/A(z) and dispersion vary axially — residuals must use
    the cross_section array rather than assuming a constant value.
    """

    natural_dims = ["axial", "radial", "angular"]

    def __init__(self, length: float, r_inlet: float, r_outlet: float):
        self.length = length
        self.r_inlet = r_inlet
        self.r_outlet = r_outlet

    def extent(self, dim: str) -> tuple[float, float]:
        if dim == "axial":
            return (0.0, self.length)
        if dim == "radial":
            r_max = max(self.r_inlet, self.r_outlet)
            return (0.0, r_max)
        if dim == "angular":
            return (0.0, 2 * np.pi)
        raise ValueError(f"Unknown dim: {dim}")

    def boundary(self, dim: str) -> DimBoundary:
        if dim == "axial":
            return DimBoundary("inlet", "outlet")
        if dim == "radial":
            return DimBoundary("symmetry", "wall")
        if dim == "angular":
            return DimBoundary("periodic", "periodic")
        raise ValueError(f"Unknown dim: {dim}")

    def spatially_varying(
        self, n_cells: dict[str, int] | None = None
    ) -> dict[str, np.ndarray]:
        if not n_cells or "axial" not in n_cells:
            return {}
        z = np.linspace(0, self.length, n_cells["axial"])
        r = self.r_inlet + (self.r_outlet - self.r_inlet) * z / self.length
        return {"cross_section": np.pi * r**2}


class Sphere(PhysicalGeometry):
    """Sphere. Used for particle phase in GRM-type models."""

    natural_dims = ["radial", "polar", "azimuthal"]

    def __init__(self, radius: float):
        self.radius = radius

    def extent(self, dim: str) -> tuple[float, float]:
        if dim == "radial":
            return (0.0, self.radius)
        if dim == "polar":
            return (0.0, np.pi)
        if dim == "azimuthal":
            return (0.0, 2 * np.pi)
        raise ValueError(f"Unknown dim: {dim}")

    def boundary(self, dim: str) -> DimBoundary:
        if dim == "radial":
            return DimBoundary("symmetry", "flux_interface")
        if dim in ("polar", "azimuthal"):
            return DimBoundary("periodic", "periodic")
        raise ValueError(f"Unknown dim: {dim}")


# ── Homogenizations ────────────────────────────────────────────────────────


class Homogenization(ABC):
    """Which dimensions of the physical geometry are spatially resolved."""

    resolved_dims: list[str]


class WellMixed(Homogenization):
    """All dimensions collapsed — 0D."""

    resolved_dims = []


class RadialAngularSymmetry(Homogenization):
    """Radial and angular symmetry — 1D axial."""

    resolved_dims = ["axial"]


class AxialAngularSymmetry(Homogenization):
    """Axial and angular symmetry — 1D radial. Used for HollowCylinder."""

    resolved_dims = ["radial"]


class AngularSymmetry(Homogenization):
    """Angular symmetry — 2D axial + radial."""

    resolved_dims = ["axial", "radial"]


class NoSymmetry(Homogenization):
    """Fully resolved — 3D."""

    resolved_dims = ["axial", "radial", "angular"]


class SphericalSymmetry(Homogenization):
    """Spherical symmetry — 1D radial."""

    resolved_dims = ["radial"]


# ── Discretized geometry ───────────────────────────────────────────────────

_FACE_ROLE = {
    "inlet": "external_port",
    "outlet": "external_port",
    "flux_interface": "flux_interface",
    "symmetry": None,
    "wall": None,
    "periodic": None,
}


class DiscretizedGeometry:
    """
    Result of combining a PhysicalGeometry with a Homogenization and a
    cell count per resolved dimension. Not instantiated by users directly —
    concrete unit operation classes build one internally.
    """

    def __init__(
        self,
        geometry: PhysicalGeometry,
        homogenization: Homogenization,
        n_cells: dict[str, int] | None = None,
    ):
        self._geometry = geometry
        self._homogenization = homogenization
        n_cells = n_cells or {}

        resolved = homogenization.resolved_dims
        missing = [d for d in resolved if d not in n_cells and d not in ("angular",)]
        if missing:
            raise ValueError(f"n_cells missing for resolved dims: {missing}")

        # shape and coordinates
        self._shape: tuple[int, ...] = tuple(n_cells[d] for d in resolved)
        self._coordinates: dict[str, np.ndarray] = {
            d: geometry.coordinates(d, n_cells[d]) for d in resolved
        }

        # spatially varying quantities (e.g. Frustum cross-section)
        self._spatially_varying = geometry.spatially_varying(n_cells)

        # derive ports from boundary types of resolved dims
        self._external_ports: dict[str, tuple] = {}
        self._flux_interfaces: dict[str, tuple] = {}
        if resolved:
            self._build_ports(resolved)
        else:
            # 0D: ports address the whole cell
            self._external_ports = {"inlet": (), "outlet": ()}

    def _build_ports(self, resolved: list[str]) -> None:
        n = len(resolved)
        for i, dim in enumerate(resolved):
            bnd = self._geometry.boundary(dim)
            prefix = (slice(None),) * i  # earlier dims: all cells

            for face_name, face_idx, face_role_str in [
                ("low", 0, bnd.low),
                ("high", -1, bnd.high),
            ]:
                role = _FACE_ROLE.get(face_role_str)
                if role is None:
                    continue

                # suffix: remaining resolved dims are fully selected
                suffix = (slice(None),) * (n - i - 1)
                idx = prefix + (face_idx,) + suffix

                port_name = face_role_str  # "inlet", "outlet", "flux_interface"

                if role == "external_port":
                    self._external_ports[port_name] = idx
                elif role == "flux_interface":
                    self._flux_interfaces[port_name] = idx

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def coordinates(self) -> dict[str, np.ndarray]:
        return self._coordinates

    @property
    def external_ports(self) -> dict[str, tuple]:
        return self._external_ports

    @property
    def flux_interfaces(self) -> dict[str, tuple]:
        return self._flux_interfaces

    @property
    def spatially_varying(self) -> dict[str, np.ndarray]:
        return self._spatially_varying

    @property
    def n_dof(self) -> int:
        return int(np.prod(self._shape)) if self._shape else 1

    def __repr__(self) -> str:
        return (
            f"DiscretizedGeometry("
            f"{self._geometry.__class__.__name__} + "
            f"{self._homogenization.__class__.__name__}, "
            f"shape={self._shape})"
        )
