# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Auxiliary state: `aux_states` property and `compute_auxiliary(t)` hook on `UnitOperationBase`; unit-op-owned aux declared via `_aux_structures`, model-owned aux declared via `model.aux_structures` and aggregated in `initialize_state()`
- `DeadEndFiltration`: `cake_height` aux entry (cake volume / membrane area), always present
- `IonicStrengthModel` stub in `auxiliary_models.py` demonstrating model-owned aux declaration
- `Column1D` stub: shows optional aux state -- `ionic_strength` only appears in output when `IonicStrengthModel` is injected
- Aux states recorded alongside primary states in `unit_solutions` with no API change to the solver

---

### Added (recording overhaul)
- `DiscretizedGeometry` -- shape, coordinates, external ports, flux interfaces, spatially varying quantities
- `State` -- geometry-aware buffer with three access lenses: `_s` (raw numpy), `state["entry"]` (numpy slice, hot path), `state.field` (dict of `xr.DataArray`, rebuilt on each access)
- `wire_to_buffer()` / `wire_states()` -- one-time setup wiring `State._s` as a shaped view into the solver's flat `y` buffer before the first section
- Named-port coupling -- `_inlet_ports`/`_outlet_ports` on unit ops; connections via `(source, dest, flow_rate)` tuples; `update_ports()` replaces the old connectivity matrix
- Geometry injection via `_make_state_geometry(state_name)` hook on `UnitOperationBase` (default: 0D `WellMixed`)
- Replay-based recording in `Solver.write_solution()`: `unit_solutions[unit][state][entry]` and `unit_derivatives[unit][state][entry]` as `xr.DataArray` with `(time, *geometry_dims, component?)` dimensions
- `time_solutions` property on `Solver` returning a flat `np.ndarray`
- `test_simulation.py` -- end-to-end integration test with output structure and physics assertions for `DeadEndFiltration`
- MyST-based documentation skeleton with Geometry, State, and System concept pages

### Fixed
- Zero-flow guard in `WeightedAverageCoupling` discarded its return value, causing division by zero and NaN when total flow rate fell below 1e-13

### Changed
- Unit op state blocks now take `DiscretizedGeometry` directly; legacy `dimensions={}` dict removed
- Solution output changed from flat numpy arrays with manual index arithmetic to nested `xr.DataArray` dicts

### Removed
- `Field` class -- superseded by `state.field` and the `xr.DataArray` recording output

## [0.0.1] - 2024-11-25

### Added
- `DiscretizedGeometry` -- shape, coordinates, external ports, flux interfaces, spatially varying quantities
- `State` -- geometry-aware buffer with three access lenses: `_s` (raw numpy), `state["entry"]` (numpy slice, hot path), `state.field` (dict of `xr.DataArray`, rebuilt on each access)
- `wire_to_buffer()` / `wire_states()` -- one-time setup wiring `State._s` as a shaped view into the solver's flat `y` buffer before the first section
- Named-port coupling -- `_inlet_ports`/`_outlet_ports` on unit ops; connections via `(source, dest, flow_rate)` tuples; `update_ports()` replaces the old connectivity matrix
- Geometry injection via `_make_state_geometry(state_name)` hook on `UnitOperationBase` (default: 0D `WellMixed`)
- Replay-based recording in `Solver.write_solution()`: `unit_solutions[unit][state][entry]` and `unit_derivatives[unit][state][entry]` as `xr.DataArray` with `(time, *geometry_dims, component?)` dimensions
- `time_solutions` property on `Solver` returning a flat `np.ndarray`
- `test_simulation.py` -- end-to-end integration test with output structure and physics assertions for `DeadEndFiltration`
- MyST-based documentation skeleton with Geometry, State, and System concept pages

### Fixed
- Zero-flow guard in `WeightedAverageCoupling` discarded its return value, causing division by zero and NaN when total flow rate fell below 1e-13

### Changed
- Unit op state blocks now take `DiscretizedGeometry` directly; legacy `dimensions={}` dict removed
- Solution output changed from flat numpy arrays with manual index arithmetic to nested `xr.DataArray` dicts

### Removed
- `Field` class -- superseded by `state.field` and the `xr.DataArray` recording output

## [0.0.1] - 2024-11-25

Initial working prototype on `master`. `DeadEndFiltration`, `Cstr`, `Inlet`, and `Outlet` unit operations with flat-index state and numpy solution arrays.
