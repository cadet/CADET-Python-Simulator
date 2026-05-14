# System and Coupling

## What is a System?

A **System** aggregates unit operations and handles the flow connections between them.
Unit operations are self-contained — they know nothing about each other. The System
is the only layer with a global view: it routes concentrations and flow rates between
unit op boundaries before each residual evaluation.

```
Solver
└─ System
     ├─ set_connections([...])   ← called once per section
     ├─ update_ports()           ← called before every residual evaluation
     └─ UnitOperation (×N)
          ├─ _inlet_ports        ← which states accept incoming flow
          ├─ _outlet_ports       ← which states provide outgoing flow
          └─ Q_in / Q_out        ← flow rates set by the System
```

## Why a connectivity matrix fails

The obvious approach — a matrix with a row per destination port and a column per origin
port — works for simple linear chains, but becomes fragile quickly:

- Ports are referenced by **integer index** (`[0, 1, 0, 0, 1e-3]` — unit 0 port 0 →
  unit 1 port 0), so adding or reordering states silently breaks connections.
- The port count on a unit op has to be computed by summing over all its internal
  state objects, which couples the system-level routing to state-level implementation
  details.
- Multi-outlet units (e.g. dead-end filter with retentate + permeate lines) require
  external bookkeeping to know which integer index maps to which physical port.

## Named ports and Connection objects

Each unit op declares two class-level dicts:

```python
class DeadEndFiltration(UnitOperationBase):
    _inlet_ports  = {"inlet": "inlet"}           # port name → state name
    _outlet_ports = {"outlet": "permeate_tank"}
```

Connections are `(source, dest, flow_rate)` tuples. When a unit has exactly one port
in the required direction the port is auto-detected; otherwise specify it explicitly:

```python
# simple chain — ports auto-detected
connections = [
    ("inlet_unit", "filter", 1e-3),
    ("filter",     "outlet_unit", 1e-3),
]

# explicit port name (e.g. filter with two outlets)
connections = [
    ("inlet_unit",        "filter",      1e-3),
    (("filter","retentate"), "retentate_collector", 0.5e-3),
    (("filter","permeate"),  "permeate_collector",  0.5e-3),
]

# vector port with index (e.g. 2D column radial slice)
connections = [
    (("col2d", "inlet", 0), ("col2d_out", "inlet"), 1e-4),
]
```

Connections are parsed into `Connection` dataclasses (source, source_port,
dest, dest_port, flow_rate, and optional index) and stored on the System for the
duration of the current time section.

## update_ports()

`update_ports()` runs before every residual evaluation. It has three steps:

**1. Reset flow rates** — all `Q_in` and `Q_out` arrays are zeroed.

**2. Accumulate Q_out** — for each connection, the source port's `Q_out` is incremented
by the connection's flow rate. A port with two downstream consumers accumulates both.

**3. Couple and set inlet states** — connections are grouped by destination port.
For each destination:
- `Q_in` is set to the total incoming flow rate.
- If `Q_in > 0`, the inlet state is updated via weighted-average coupling over all
  incoming streams.

```python
# FlowSystem uses weighted-average concentration coupling:
system.set_connections([("inlet", "cstr", 1e-3)])
# Before every residual call:
system.update_ports()
# → cstr.Q_in[0]   = 1e-3
# → cstr.states["inlet"]["c"] = inlet.states["outlet"]["c"]
```

## Coupling modes

Coupling logic lives in `coupled_state_func()`, which is called by `update_ports()` with
a list of `(port_state_dict, flow_rate)` pairs. The base implementation delegates to
`coupling_state_structure`, a dict of entry name → `CouplingInterface`:

```python
# FlowSystem
coupling_state_structure = {"c": WeightedAverageCoupling()}
```

A future `PressureSystem` would override `coupled_state_func` to implement fixed-pressure
coupling where flow rate is an algebraic unknown rather than a given. Unit operations
are unaffected by the coupling mode — they only read `Q_in`, `Q_out`, and their own
inlet state after `update_ports()` has run.

## Invariants

- `update_ports()` is always called before the first residual evaluation of each section.
- Unit ops never call port methods on other unit ops — only the System does.
- `Q_in` / `Q_out` are reset to zero at the start of every `update_ports()` call;
  they are never accumulated across sections.
