from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from CADETPythonSimulator.geometry import (
    Cylinder,
    DiscretizedGeometry,
    WellMixed,
)


class State:
    """
    Manages the state of a unit operation domain.

    The underlying array ``_s`` is the single source of truth. It is always
    mutated in-place — never replaced — so all derived views (entry slices)
    remain valid across solver steps.

    Parameters
    ----------
    name : str
        Identifier for this state block (e.g. ``"bulk"``, ``"cake"``).
    entries : dict[str, int]
        Mapping of entry name to number of values per cell.
        E.g. ``{"c": 3, "V": 1}`` for 3 components + volume.
    geometry : DiscretizedGeometry
        Spatial structure. Carries shape, coordinates, and port locations.
    """

    def __init__(
        self,
        name: str,
        entries: dict[str, int],
        geometry: DiscretizedGeometry,
    ) -> None:
        self.name = name
        self.entries = entries
        self._geometry = geometry

        # ── allocate s (never replaced after this point) ───────────────────
        n_entries_total = sum(entries.values())
        self._s = np.zeros((*self._geometry.shape, n_entries_total))

        # pre-compute entry slices into the last axis
        self._entry_slices: dict[str, slice] = {}
        i = 0
        for entry, n in entries.items():
            self._entry_slices[entry] = slice(i, i + n)
            i += n

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
        After this, ``field`` returns DataArrays wrapping the new memory.
        """
        self._s = buf[offset : offset + self.n_dof].reshape(self._s.shape)

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
        return self._build_field()

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
            entry: self._s[(*idx, self._entry_slices[entry])] for entry in self.entries
        }

    def set_port_state(self, port_name: str, values: dict[str, np.ndarray]) -> None:
        """Set entry values at the named external port (in-place)."""
        idx = self._geometry.external_ports[port_name]
        for entry, value in values.items():
            self._s[(*idx, self._entry_slices[entry])] = value

    # ── dunder ────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return (
            f"State(name='{self.name}', "
            f"geometry={self._geometry!r}, "
            f"entries={self.entries})"
        )


# ── factory ────────────────────────────────────────────────────────────────


def state_factory(
    instance: Any,
    name: str,
    entries: dict[str, int | str],
    geometry: DiscretizedGeometry | None = None,
) -> State:
    """
    Create a State, resolving string attribute references on ``instance``.

    If no geometry is provided, defaults to a 0-D WellMixed geometry.
    """
    resolved_entries = {
        entry: (n if isinstance(n, int) else getattr(instance, n))
        for entry, n in entries.items()
    }
    if geometry is None:
        geometry = DiscretizedGeometry(Cylinder(length=1.0), WellMixed())
    return State(name=name, entries=resolved_entries, geometry=geometry)
