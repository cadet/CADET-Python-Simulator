from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from CADETPythonSimulator.geometry import DiscretizedGeometry


class State:
    """
    Manages the state of a unit operation domain.

    The underlying array ``s`` is the single source of truth. It is always
    mutated in-place — never replaced — so all derived views (entry slices,
    xarray DataArrays) remain valid across solver steps.

    Parameters
    ----------
    name : str
        Identifier for this state block (e.g. ``"bulk"``, ``"cake"``).
    entries : dict[str, int]
        Mapping of entry name to number of values per cell.
        E.g. ``{"c": 3, "V": 1}`` for 3 components + volume.
    geometry : DiscretizedGeometry, optional
        Spatial structure. If omitted, a 0-D anonymous geometry is used
        (backward-compatible with the old ``dimensions`` dict interface).
    """

    def __init__(
        self,
        name: str,
        entries: dict[str, int],
        geometry: DiscretizedGeometry | None = None,
        # ── legacy params kept for backward compatibility ──────────────────
        dimensions: dict[str, int] | None = None,
        n_inlet_ports: int = 0,
        n_outlet_ports: int = 0,
    ) -> None:
        self.name = name
        self.entries = entries

        if geometry is not None:
            self._geometry = geometry
            self._legacy_ports = None
        else:
            dims = dimensions or {}
            # resolve string port counts (e.g. n_inlet_ports="radial" → dims["radial"])
            inlet_mapping = n_inlet_ports if isinstance(n_inlet_ports, str) else None
            outlet_mapping = n_outlet_ports if isinstance(n_outlet_ports, str) else None
            n_inlet = dims[n_inlet_ports] if isinstance(n_inlet_ports, str) else n_inlet_ports
            n_outlet = dims[n_outlet_ports] if isinstance(n_outlet_ports, str) else n_outlet_ports

            self._geometry = _geometry_from_legacy(dims, n_inlet_ports, n_outlet_ports)
            self._legacy_ports = {
                "dimensions": dims,
                "n_inlet_ports": n_inlet,
                "n_outlet_ports": n_outlet,
                "inlet_mapping": inlet_mapping,
                "outlet_mapping": outlet_mapping,
            }

        # ── allocate s (never replaced after this point) ───────────────────
        n_entries_total = sum(entries.values())
        self._s = np.zeros((*self._geometry.shape, n_entries_total))

        # pre-compute entry slices into the last axis
        self._entry_slices: dict[str, slice] = {}
        i = 0
        for entry, n in entries.items():
            self._entry_slices[entry] = slice(i, i + n)
            i += n

        # xarray views — created once, always current via shared memory
        self._field: dict[str, xr.DataArray] = self._build_field()

    # ── geometry access ────────────────────────────────────────────────────

    @property
    def geometry(self) -> DiscretizedGeometry:
        return self._geometry

    # ── shape / dof ───────────────────────────────────────────────────────

    @property
    def dimension_shape(self) -> tuple[int, ...]:
        return self._geometry.shape

    @property
    def n_dimensions(self) -> int:
        return len(self._geometry.shape)

    @property
    def n_cells(self) -> int:
        return self._geometry.n_dof

    @property
    def n_entries(self) -> int:
        return sum(self.entries.values())

    @property
    def shape(self) -> tuple[int, ...]:
        return self._s.shape

    @property
    def n_dof(self) -> int:
        return int(np.prod(self._s.shape))

    # ── buffer wiring (one-time setup before solve) ───────────────────────

    def wire_to_buffer(self, buf: np.ndarray, offset: int) -> None:
        """Replace _s with a shaped view into buf[offset:offset+n_dof].

        Called once by the system before the first solver section.
        Rebuilds field DataArrays so they point at the new memory.
        """
        self._s = buf[offset : offset + self.n_dof].reshape(self._s.shape)
        self._field = self._build_field()

    # ── s access (in-place invariant) ─────────────────────────────────────

    @property
    def s(self) -> np.ndarray:
        return self._s

    @s.setter
    def s(self, value: np.ndarray) -> None:
        self._s[...] = value

    @property
    def s_flat(self) -> np.ndarray:
        return self._s.reshape(-1)

    @s_flat.setter
    def s_flat(self, value: np.ndarray) -> None:
        self._s[...] = np.asarray(value).reshape(self._s.shape)

    @property
    def s_split(self) -> dict[str, np.ndarray]:
        return {entry: self[entry] for entry in self.entries}

    @s_split.setter
    def s_split(self, s_split: dict[str, np.ndarray]) -> None:
        for key, value in s_split.items():
            self[key][:] = value

    # ── entry access ──────────────────────────────────────────────────────

    def __getitem__(self, entry: str) -> np.ndarray:
        try:
            return self._s[..., self._entry_slices[entry]]
        except KeyError:
            raise KeyError(f"'{entry}' is not a valid state entry.")

    def __setitem__(self, entry: str, value: np.ndarray) -> None:
        if entry not in self._entry_slices:
            raise KeyError(f"'{entry}' is not a valid state entry.")
        self._s[..., self._entry_slices[entry]] = value

    # ── xarray field (zero-copy views) ────────────────────────────────────

    @property
    def field(self) -> dict[str, xr.DataArray]:
        """Zero-copy xarray views per entry. Always reflects current ``s``."""
        return self._field

    def _build_field(self) -> dict[str, xr.DataArray]:
        coords = self._geometry.coordinates
        spatial_dims = list(coords.keys())
        result = {}
        for entry, sl in self._entry_slices.items():
            data = self._s[..., sl]
            n = self.entries[entry]
            if n > 1:
                da = xr.DataArray(
                    data,
                    coords={**coords, "component": np.arange(n)},
                    dims=spatial_dims + ["component"],
                    name=entry,
                )
            else:
                da = xr.DataArray(
                    data[..., 0],
                    coords=coords,
                    dims=spatial_dims,
                    name=entry,
                )
            result[entry] = da
        return result

    # ── port access ───────────────────────────────────────────────────────

    def get_port_state(self, port_name: str) -> dict[str, np.ndarray]:
        """Return entry arrays at the named external port (views into s)."""
        idx = self._geometry.external_ports[port_name]
        return {
            entry: self._s[(*idx, self._entry_slices[entry])]
            for entry in self.entries
        }

    def set_port_state(self, port_name: str, values: dict[str, np.ndarray]) -> None:
        """Set entry values at the named external port (in-place)."""
        idx = self._geometry.external_ports[port_name]
        for entry, value in values.items():
            self._s[(*idx, self._entry_slices[entry])] = value

    # ── legacy port interface (backward compatibility) ─────────────────────

    @property
    def n_inlet_ports(self) -> int:
        if self._legacy_ports is not None:
            return self._legacy_ports["n_inlet_ports"]
        return sum(1 for k in self._geometry.external_ports if k == "inlet")

    @property
    def n_outlet_ports(self) -> int:
        if self._legacy_ports is not None:
            return self._legacy_ports["n_outlet_ports"]
        return sum(1 for k in self._geometry.external_ports if k == "outlet")

    @property
    def inlet_port_mapping(self) -> str | None:
        if self._legacy_ports is None:
            return None
        return self._legacy_ports.get("inlet_mapping")

    @property
    def outlet_port_mapping(self) -> str | None:
        if self._legacy_ports is None:
            return None
        return self._legacy_ports.get("outlet_mapping")

    def set_inlet_port_state(
        self, inlet_port_state: dict[str, np.ndarray], port_index: int
    ) -> None:
        if port_index > (self.n_inlet_ports - 1):
            raise ValueError("Port index exceeds number of inlet ports.")
        idx = _legacy_port_slice(self, "inlet", port_index)
        s_split = self.s_split
        for entry, value in inlet_port_state.items():
            s_split[entry][idx] = value

    def get_outlet_port_state(self, port_index: int) -> dict[str, np.ndarray]:
        if port_index > (self.n_outlet_ports - 1):
            raise ValueError("Port index exceeds number of outlet ports.")
        idx = _legacy_port_slice(self, "outlet", port_index)
        return {entry: self.s_split[entry][idx] for entry in self.entries}

    def validate_coupling_structure(self, coupling_structure: dict[str, int]) -> bool:
        flag = True
        for entry, n_entries in coupling_structure:
            if entry not in self.entries:
                import warnings
                warnings.warn(f"Entry '{entry}' is not found in entries.")
                flag = False
            elif n_entries != self.entries[entry]:
                import warnings
                warnings.warn(
                    f"Entry '{entry}' does not have shape {self.entries[entry]}."
                )
                flag = False
        return flag

    # ── dunder ────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"State(name='{self.name}', "
            f"geometry={self._geometry!r}, "
            f"entries={self.entries})"
        )


# ── helpers ────────────────────────────────────────────────────────────────

def _legacy_port_slice(
    state: State, port_type: str, port_index: int
) -> tuple:
    """Build a slice tuple for legacy port access (used by old unit ops)."""
    n_dims = len(state.dimension_shape)

    if port_type == "inlet":
        mapping = state.inlet_port_mapping
        face_index = 0
    else:
        mapping = state.outlet_port_mapping
        face_index = -1

    slice_indices = [face_index] * (n_dims + 1)

    if mapping is not None:
        dim_keys = list(state._legacy_ports["dimensions"].keys())
        port_dim_index = dim_keys.index(mapping)
        slice_indices[port_dim_index] = port_index

    slice_indices[-1] = slice(None)
    return tuple(slice_indices)


def _geometry_from_legacy(
    dimensions: dict[str, int],
    n_inlet_ports: int | str,
    n_outlet_ports: int | str,
) -> DiscretizedGeometry:
    """
    Build a minimal DiscretizedGeometry from an old-style dimensions dict.
    Physical parameters (length, radius) are placeholders — this is only for
    structural compatibility until unit ops are upgraded to pass geometry directly.
    """
    from CADETPythonSimulator.geometry import (
        AngularSymmetry,
        Cylinder,
        RadialAngularSymmetry,
        Sphere,
        SphericalSymmetry,
        WellMixed,
    )

    if not dimensions:
        return DiscretizedGeometry(Cylinder(length=1.0), WellMixed())

    # normalise: strip optional "n_" prefix for matching
    n_cells = {k.removeprefix("n_"): v for k, v in dimensions.items()}
    normalized = list(n_cells.keys())

    # best-effort mapping of legacy dimension names
    if normalized == ["axial"]:
        return DiscretizedGeometry(
            Cylinder(length=1.0), RadialAngularSymmetry(), {"axial": n_cells["axial"]}
        )
    if normalized == ["axial", "radial"]:
        return DiscretizedGeometry(
            Cylinder(length=1.0, radius=1.0),
            AngularSymmetry(),
            {"axial": n_cells["axial"], "radial": n_cells["radial"]},
        )
    if normalized == ["radial"]:
        return DiscretizedGeometry(
            Sphere(radius=1.0), SphericalSymmetry(), {"radial": n_cells["radial"]}
        )

    # fallback: treat as 0D
    return DiscretizedGeometry(Cylinder(length=1.0), WellMixed())


# ── factory ────────────────────────────────────────────────────────────────

def state_factory(
    instance: Any,
    name: str,
    dimensions: tuple[str, ...] | dict,
    entries: dict[str, int | str],
    n_inlet_ports: int | str = 0,
    n_outlet_ports: int | str = 0,
    geometry: DiscretizedGeometry | None = None,
) -> State:
    """
    Create a State, resolving string attribute references on ``instance``.

    Accepts either a ``geometry`` (new interface) or the legacy
    ``dimensions`` tuple / port arguments (old interface).
    """
    # resolve entries
    resolved_entries = {
        entry: (n if isinstance(n, int) else getattr(instance, n))
        for entry, n in entries.items()
    }

    if geometry is not None:
        return State(name=name, entries=resolved_entries, geometry=geometry)

    # legacy path: resolve dimensions from instance attributes
    if isinstance(dimensions, (list, tuple)):
        resolved_dims = {dim: getattr(instance, dim) for dim in dimensions}
    else:
        resolved_dims = dict(dimensions)

    # resolve port counts that reference dimension names
    resolved_n_inlet = (
        resolved_dims[n_inlet_ports]
        if isinstance(n_inlet_ports, str)
        else n_inlet_ports
    )
    resolved_n_outlet = (
        resolved_dims[n_outlet_ports]
        if isinstance(n_outlet_ports, str)
        else n_outlet_ports
    )

    return State(
        name=name,
        entries=resolved_entries,
        dimensions=resolved_dims,
        n_inlet_ports=resolved_n_inlet,
        n_outlet_ports=resolved_n_outlet,
    )
