import pytest
import numpy as np
from CADETPythonSimulator.state import State


@pytest.mark.parametrize(
    "state_config, expected",
    [
        (
            {
                "name": "outlet",
                "dimensions": {},
                "entries": {"c": 2, "viscosity": 1},
                "n_outlet_ports": 1,
            },
            {
                "dimension_shape": (),
                "n_dimensions": 0,
                "n_cells": 1,
                "n_entries": 3,
                "shape": (3,),
                "n_dof": 3,
                "n_inlet_ports": 0,
                "inlet_port_mapping": None,
                "n_outlet_ports": 1,
                "outlet_port_mapping": None,
                "s_split": {
                    "c": {"slice": np.s_[:], "value": [0, 1]},
                    "viscosity": {"slice": np.s_[:], "value": [2]},
                },
                "outlet_state": {0: {"c": [0, 1], "viscosity": [2]}},
            },
        ),
        (
            {
                "name": "inlet",
                "dimensions": {},
                "entries": {"c": 2, "viscosity": 1},
                "n_inlet_ports": 1,
            },
            {
                "dimension_shape": (),
                "n_dimensions": 0,
                "n_cells": 1,
                "n_entries": 3,
                "shape": (3,),
                "n_dof": 3,
                "n_inlet_ports": 1,
                "inlet_port_mapping": None,
                "n_outlet_ports": 0,
                "outlet_port_mapping": None,
                "s_split": {
                    "c": {"slice": np.s_[:], "value": [0, 1]},
                    "viscosity": {"slice": np.s_[:], "value": [2]},
                },
                "inlet_state": {0: {"slice": np.s_[:], "value": [0.1, 0.2, 0.3]}},
            },
        ),
        (
            {
                "name": "permeate",
                "dimensions": {"axial": 10},
                "entries": {"c": 2, "viscosity": 1},
                "n_outlet_ports": 1,
            },
            {
                "dimension_shape": (10,),
                "n_dimensions": 1,
                "n_cells": 10,
                "n_entries": 3,
                "shape": (10, 3),
                "n_dof": 30,
                "n_inlet_ports": 0,
                "inlet_port_mapping": None,
                "n_outlet_ports": 1,
                "outlet_port_mapping": None,
                "s_split": {
                    "c": {
                        "slice": np.s_[:],
                        "value": [
                            [0, 1],
                            [3, 4],
                            [6, 7],
                            [9, 10],
                            [12, 13],
                            [15, 16],
                            [18, 19],
                            [21, 22],
                            [24, 25],
                            [27, 28],
                        ],
                    },
                    "viscosity": {
                        "slice": np.s_[:],
                        "value": [
                            [2],
                            [5],
                            [8],
                            [11],
                            [14],
                            [17],
                            [20],
                            [23],
                            [26],
                            [29],
                        ],
                    },
                },
                "outlet_state": {0: {"c": [27, 28], "viscosity": [29]}},
            },
        ),
        (
            {
                "name": "bulk",
                "dimensions": {"axial": 10},
                "entries": {"c": 2, "viscosity": 1},
                "n_inlet_ports": 1,
                "n_outlet_ports": 1,
            },
            {
                "dimension_shape": (10,),
                "n_dimensions": 1,
                "n_cells": 10,
                "n_entries": 3,
                "shape": (10, 3),
                "n_dof": 30,
                "n_inlet_ports": 1,
                "inlet_port_mapping": None,
                "n_outlet_ports": 1,
                "outlet_port_mapping": None,
                "s_split": {
                    "c": {
                        "slice": np.s_[:],
                        "value": [
                            [0, 1],
                            [3, 4],
                            [6, 7],
                            [9, 10],
                            [12, 13],
                            [15, 16],
                            [18, 19],
                            [21, 22],
                            [24, 25],
                            [27, 28],
                        ],
                    },
                    "viscosity": {
                        "slice": np.s_[:],
                        "value": [
                            [2],
                            [5],
                            [8],
                            [11],
                            [14],
                            [17],
                            [20],
                            [23],
                            [26],
                            [29],
                        ],
                    },
                },
                "inlet_state": {
                    0: {
                        "slice": np.s_[:],
                        "value": [
                            [0.1, 0.2, 0.3],
                            [3, 4, 5],
                            [6, 7, 8],
                            [9, 10, 11],
                            [12, 13, 14],
                            [15, 16, 17],
                            [18, 19, 20],
                            [21, 22, 23],
                            [24, 25, 26],
                            [27, 28, 29],
                        ],
                    }
                },
                "outlet_state": {0: {"c": [27, 28], "viscosity": [29]}},
            },
        ),
        (
            {
                "name": "_2d_bulk",
                "dimensions": {"axial": 10, "radial": 3},
                "entries": {"c": 2, "viscosity": 1},
                "n_inlet_ports": "radial",
                "n_outlet_ports": "radial",
            },
            {
                "dimension_shape": (10, 3),
                "n_dimensions": 2,
                "n_cells": 30,
                "n_entries": 3,
                "shape": (10, 3, 3),
                "n_dof": 90,
                "n_inlet_ports": 3,
                "inlet_port_mapping": "radial",
                "n_outlet_ports": 3,
                "outlet_port_mapping": "radial",
                "s_split": {
                    "c": {
                        "slice": np.s_[0:3, 0:3, :],
                        "value": [
                            [[0, 1], [3, 4], [6, 7]],
                            [[9, 10], [12, 13], [15, 16]],
                            [[18, 19], [21, 22], [24, 25]],
                        ],
                    },
                    "viscosity": {
                        "slice": np.s_[0:3, 0:3, :],
                        "value": [
                            [[2], [5], [8]],
                            [[11], [14], [17]],
                            [[20], [23], [26]],
                        ],
                    },
                },
                "inlet_state": {
                    0: {
                        "slice": np.s_[0:3, 0:3, :],
                        "value": [
                            [[0.1, 0.2, 0.3], [3, 4, 5], [6, 7, 8]],
                            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
                        ],
                    },
                    1: {
                        "slice": np.s_[0:3, 0:3, :],
                        "value": [
                            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [6, 7, 8]],
                            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
                        ],
                    },
                    2: {
                        "slice": np.s_[0:3, 0:3, :],
                        "value": [
                            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
                            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
                        ],
                    },
                },
                "outlet_state": {
                    0: {"c": [81, 82], "viscosity": [83]},
                    1: {"c": [84, 85], "viscosity": [86]},
                    2: {"c": [87, 88], "viscosity": [89]},
                },
            },
        ),
    ],
)
class TestState:
    """Test state class."""

    def test_state_dimensions(self, state_config, expected):
        """Test if state parameters are set correctly."""
        state = State(**state_config)
        assert state.dimension_shape == expected["dimension_shape"]
        assert state.n_dimensions == expected["n_dimensions"]
        assert state.n_cells == expected["n_cells"]
        assert state.n_entries == expected["n_entries"]
        assert state.shape == expected["shape"]
        assert state.n_dof == expected["n_dof"]
        assert state.n_inlet_ports == expected["n_inlet_ports"]
        assert state.inlet_port_mapping == expected["inlet_port_mapping"]
        assert state.n_outlet_ports == expected["n_outlet_ports"]
        assert state.outlet_port_mapping == expected["outlet_port_mapping"]

    def test_s_split(self, state_config, expected):
        """Tests if state spltitting works as intended."""
        state = State(**state_config)

        new_state = np.arange(state.n_dof, dtype=float).reshape(state.shape)
        state.s = new_state

        np.testing.assert_array_equal(state.s_flat, new_state.reshape(-1))

        s_split = state.s_split
        for entry, slice_information in expected["s_split"].items():
            split_slice = s_split[entry][slice_information["slice"]]
            np.testing.assert_array_equal(split_slice, slice_information["value"])

        s_in = {"c": [0.1, 0.2], "viscosity": [0.3]}
        if state.n_inlet_ports == 0:
            with pytest.raises(Exception):
                state.set_inlet_port_state(s_in, 0)
        else:
            for port in range(state.n_inlet_ports):
                state.set_inlet_port_state(s_in, port)

                slice_information = expected["inlet_state"][port]
                state_slice = state.s[slice_information["slice"]]
                np.testing.assert_array_equal(state_slice, slice_information["value"])

        if state.n_outlet_ports == 0:
            with pytest.raises(Exception):
                state.get_outlet_port_state(0)
        else:
            for port in range(state.n_outlet_ports):
                s_out = state.get_outlet_port_state(port)
                np.testing.assert_equal(s_out, expected["outlet_state"][port])


# %% Test coupling structure
# TODO: Add Test for coupling structure
# TODO: Add Test for setting state if not all entries are in coupling structure

# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_state.py"])
