# State

## What is a State?

Each unit operation manages one or more **State** objects — named blocks of numerical
variables that the DAE solver tracks over time. A chromatography column might have a `bulk`
state (concentration profile along the column) and a `particle` state (concentration inside
the particles). A dead-end filtration unit has `cake`, `permeate_tank`, and `inlet` states.

A `State` has two orthogonal aspects:

- **Geometry** — the spatial structure: how many cells, what coordinates, which faces are ports
- **Entries** — what variables live at each cell: `{"c": 3, "V": 1}` for 3 components plus volume

## The single source of truth

Every `State` owns one numpy array, `_s`, of shape `(*geometry.shape, n_entries)`:

```
geometry.shape = (10,)   →   _s.shape = (10, 4)   for entries {"c": 3, "V": 1}
geometry.shape = (5, 3)  →   _s.shape = (5, 3, 2)  for entries {"c": 2}
geometry.shape = ()      →   _s.shape = (3,)        for entries {"c": 2, "V": 1}
```

`_s` is **never replaced after `__init__`** — only mutated in-place. This invariant ensures
that all derived views (numpy slices, xarray DataArrays) remain valid across solver steps.
The solver's `y` buffer can be wired in as a view of `_s`, so the solver and the state share
the same memory without copying.

## Three lenses on one buffer

```python
state._s                    # raw numpy array — source of truth
state["c"]                  # numpy view slice — hot path in residuals
state.field["c"]            # xr.DataArray — labeled access, always current
```

### `state["c"]` — numpy slice

Returns a view into `_s` for the named entry. Fast, zero-copy, suitable for residual
functions that run thousands of times per solve.

```python
bulk = State(name="bulk", entries={"c": 3, "V": 1}, geometry=g)
bulk["c"]          # shape (n_axial, 3) — view, not a copy
bulk["c"][:] = 0   # mutates _s in place
```

### `state.field["c"]` — xr.DataArray view

`state.field` calls `_build_field()` on every access and returns a fresh dict of
zero-copy `xr.DataArray` wrappers over the current `_s` memory. Because there is no cached
`_field` attribute, there is no stale-reference risk across `wire_to_buffer()`.

```python
bulk.field["c"]                        # xr.DataArray, dims=["axial", "component"]
bulk.field["c"].sel(component=0)       # first component, still a view into _s
bulk.field["V"]                        # xr.DataArray, dims=["axial"]  (scalar entry, no component dim)
```

Multi-component entries get a `"component"` dimension. Single-component entries are squeezed
— no trailing size-1 dim.

:::{note}
`state.field` is for **reading** labeled data and for output routines.
For residual computation, use `state["entry"]` — it avoids the `_build_field()` overhead on the hot path.
:::

## Entries

`entries` is a dict mapping entry name to number of values per cell:

```python
entries = {"c": 3, "V": 1}   # 3 components, 1 volume per cell
```

Entries are packed into the last axis of `_s` in order. Entry slices are pre-computed at
`__init__` time:

```
_s[..., 0:3]  →  "c"
_s[..., 3:4]  →  "V"
```

## Geometry

`State` takes a `DiscretizedGeometry` that carries the shape, coordinates, and port locations.
See [Geometry](geometry.md) for details on how to build one.

```python
from CADETPythonSimulator.geometry import Cylinder, RadialAngularSymmetry, DiscretizedGeometry
from CADETPythonSimulator.state import State

g = DiscretizedGeometry(Cylinder(length=0.1), RadialAngularSymmetry(), {"axial": 20})
bulk = State(name="bulk", entries={"c": 3, "V": 1}, geometry=g)

bulk.shape            # (20, 4)
bulk.n_dof            # 80
bulk.dimension_shape  # (20,)
```

## Ports

Ports are boundary faces of the geometry exposed to the `System` for flow routing.
`get_port_state` and `set_port_state` take a port name as defined by the geometry:

```python
inlet_values = bulk.get_port_state("inlet")   # {"c": array([...]), "V": array([...])}
bulk.set_port_state("inlet", {"c": ..., "V": ...})
```

For a 1D column, `"inlet"` maps to cell `0` and `"outlet"` to cell `-1`.
For a 2D column, `"inlet"` maps to `s[0, :]` — a full radial slice.

The `System` calls these methods during `update_ports()` before each residual evaluation.
Unit operations only read from their own ports; they never call port methods on other units.

## Invariants

- `state._s` is never replaced — all setters use `self._s[...] = value`
- `state["entry"]` returns a numpy **view**, not a copy
- `state.field["entry"]` is an xr.DataArray zero-copy **view** into `_s`, rebuilt on each `field` access
- Port access (`get_port_state`, `set_port_state`) indexes directly into `_s`

## Solver integration

During a solve, the solver owns a flat buffer `y`. The `State._s` array can be made a
**shaped view into `y`** so the solver and state share the same memory:

```
y (flat, owned by IDA)
└─ _s = y[offset:offset+n_dof].reshape(*geometry.shape, n_entries)
   ├─ state["c"]        numpy slice — residual functions
   └─ state.field["c"]  xr.DataArray — labeled access
```

At `write_solution()` time, the solver replays each recorded time step by setting
`system.y` to the corresponding row of `y_history`. This updates all wired `_s` views.
For each state, `state.field[entry].values.copy()` captures an owned snapshot.
After all steps the snapshots are stacked into `xr.DataArray` objects with a `time` dim
prepended — shape `(T, *geometry_dims, n_comp?)`. The live state buffer is always numpy;
xarray only appears at the output boundary.
