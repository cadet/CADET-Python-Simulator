import numpy as np
import pytest

from CADETPythonSimulator.system import FlowSystem, SystemBase
from conftest import (
    SystemFixture,
    InletFixture,
    OutletFixture,
    CstrFixture,
    CrossFlowFiltrationFixture,
)


# ── Systems for parametrized connectivity tests ────────────────────────────────
# Each topology gets its own instance; they are mutated by test_coupling.

_split_system = FlowSystem([InletFixture(), CstrFixture(), OutletFixture()])
_split_system.initialize_state()

_merge_system = FlowSystem([
    InletFixture(name="inlet_a"),
    InletFixture(name="inlet_b"),
    CstrFixture(),
    OutletFixture(),
])
_merge_system.initialize_state()

_recycle_system = FlowSystem([InletFixture(), CstrFixture(), OutletFixture()])
_recycle_system.initialize_state()

_multiport_system = FlowSystem([
    InletFixture(),
    CrossFlowFiltrationFixture(),
    OutletFixture(name="retentate_out"),
    OutletFixture(name="permeate_out"),
])
_multiport_system.initialize_state()


# ── Structure ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "system, expected",
    [
        (SystemFixture(), {"n_dof": 29}),
        (_split_system, {"n_dof": 9}),
        (_merge_system, {"n_dof": 11}),
        (_multiport_system, {"n_dof": 86}),
    ],
)
class TestSystem:
    def test_system_structure(self, system, expected):
        assert system.n_dof == expected["n_dof"]


# ── Connectivity (port coupling + flow-rate accumulation) ──────────────────────

@pytest.mark.parametrize(
    "system, connections, expected_state",
    [
        # simple linear chain: inlet → DEF → outlet
        (
            SystemFixture(),
            [
                ("inlet", "dead_end_filtration", 1e-3),
                ("dead_end_filtration", "outlet", 1e-3),
            ],
            [
                0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 22, 23,
            ],
        ),
        # split: one source, two destinations — Q_out accumulates
        (
            _split_system,
            [
                ("inlet", "cstr", 0.6e-3),
                ("inlet", "outlet", 0.4e-3),
            ],
            # inlet.outlet.c=[0,1] copied to both cstr.inlet and outlet.inlet
            [0, 1, 0, 1, 4, 5, 6, 0, 1],
        ),
        # merge: two sources combined by weighted average at cstr inlet
        (
            _merge_system,
            [
                ("inlet_a", "cstr", 0.3e-3),
                ("inlet_b", "cstr", 0.7e-3),
                ("cstr", "outlet", 1.0e-3),
            ],
            # cstr.inlet.c = (0.3*[0,1] + 0.7*[2,3]) / 1.0 = [1.4, 2.4]
            # outlet.inlet.c = cstr.bulk.c = [6, 7]
            [0, 1, 2, 3, 1.4, 2.4, 6, 7, 8, 6, 7],
        ),
        # recycle: 10% of cstr outlet feeds back into cstr inlet
        (
            _recycle_system,
            [
                ("inlet", "cstr", 1e-3),
                ("cstr", "outlet", 0.9e-3),
                ("cstr", "cstr", 0.1e-3),
            ],
            # cstr.inlet.c = (1e-3*[0,1] + 0.1e-3*[4,5]) / 1.1e-3 = [4/11, 15/11]
            # outlet.inlet.c = cstr.bulk.c (pre-update) = [4, 5]
            [0, 1, 4 / 11, 15 / 11, 4, 5, 6, 4, 5],
        ),
        # multiple outlet ports: explicit (unit, port) spec for CFF
        (
            _multiport_system,
            [
                ("inlet", "cross_flow_filtration", 1e-3),
                (("cross_flow_filtration", "retentate"), "retentate_out", 0.5e-3),
                (("cross_flow_filtration", "permeate"), "permeate_out", 0.5e-3),
            ],
            # cff.retentate[0, c] ← inlet.c = [0, 1]  (system DOFs 2-3)
            # retentate_out.inlet.c ← cff.retentate[-1, c] = [38, 39]
            # permeate_out.inlet.c  ← cff.permeate[-1, c]  = [78, 79]
            [*range(2), 0, 1, *range(4, 82), 38, 39, 78, 79],
        ),
    ],
)
class TestSystemConnectivity:
    def test_coupling(self, system, connections, expected_state):
        y = np.arange(system.n_dof, dtype=float)
        system.y = y
        system.y_dot = 2 * y
        system.set_connections(connections)
        system.update_ports()
        np.testing.assert_almost_equal(system.y, expected_state)


# ── Initial value setup ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "system, connections, initial_values, expected_state",
    [
        (
            SystemFixture(),
            [
                ("inlet", "dead_end_filtration", 1e-3),
                ("dead_end_filtration", "outlet", 1e-3),
            ],
            [
                0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 22, 23,
            ],
            [
                0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 22, 23,
            ],
        )
    ],
)
class TestSystemInitializeInitialValue:
    def test_initialize(self, system, connections, initial_values, expected_state):
        system.y = initial_values
        system.set_connections(connections)


if __name__ == "__main__":
    pytest.main(["test_system.py"])
