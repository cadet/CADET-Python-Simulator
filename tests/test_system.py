import numpy as np
import pytest

from test_solver import SystemFixture
from CADETPythonSimulator.system import SystemBase


@pytest.mark.parametrize(
    "system, expected",
    [
        (
            SystemFixture(),
            {
                "n_dof": 29,
                "unit_slices": {
                    "inlet": slice(0, 2, None),
                    "dead_end_filtration": slice(2, 11, None),
                    "outlet": slice(11, 13, None),
                },
                "unit_solution": {
                    "inlet": {
                        "c_out": np.array([[0.0, 1.0], [0.0, 2.0]]),
                        "c_out_dot": np.array([[0.0, 2.0], [0.0, 4.0]]),
                    },
                    "dead_end_filtration": {
                        "c_in": np.array([[2.0, 3.0], [4.0, 6.0]]),
                        "c_in_dot": np.array([[4.0, 6.0], [8.0, 12.0]]),
                        "Vp": np.array([[4.0], [8.0]]),
                        "Vp_dot": np.array([[8.0], [16.0]]),
                        "Rc": np.array([[5.0], [10.0]]),
                        "Rc_dot": np.array([[10.0], [20.0]]),
                        "mc": np.array([[6.0, 7.0], [12.0, 14.0]]),
                        "mc_dot": np.array([[12.0, 14.0], [24.0, 28.0]]),
                        "c_out": np.array([[8.0, 9.0], [16.0, 18.0]]),
                        "c_out_dot": np.array([[16.0, 18.0], [32.0, 36.0]]),
                    },
                    "outlet": {
                        "c_in": np.array([[10.0, 11.0], [20.0, 22.0]]),
                        "c_in_dot": np.array([[20.0, 22.0], [40.0, 44.0]]),
                    },
                },
                "section_states": {
                    0: {
                        "parameters_start": {
                            "inlet": {"c": np.array([1.0, 2.0])},
                            "dead_end_filtration": {},
                            "outlet": {},
                        },
                        "parameters_end": {
                            "inlet": {"c": np.array([1.0, 2.0])},
                            "dead_end_filtration": {},
                            "outlet": {},
                        },
                    },
                    1: {
                        "parameters_start": {
                            "inlet": {"c": np.array([0.0, 1.0])},
                            "dead_end_filtration": {},
                            "outlet": {},
                        },
                        "parameters_end": {
                            "inlet": {"c": np.array([2.0, 1.0])},
                            "dead_end_filtration": {},
                            "outlet": {},
                        },
                    },
                },
            },
        )
    ],
)
class TestSystem:
    """Test Class for system."""

    def test_system_structure(self, system, expected):
        """Test system structure."""
        assert system.n_dof == expected["n_dof"]


# %% System Connectivity


@pytest.mark.parametrize(
    "system, connections, expected_matrix, expected_state",
    [
        (
            SystemFixture(),
            [[0, 1, 0, 0, 1e-3], [1, 2, 0, 0, 1e-3]],
            [[0.001, 0.0], [0.0, 0.001]],
            [
                0,
                1,
                0,
                1,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                22,
                23,
            ],
        )
    ],
    # TODO: Add tests with split/combined streams.
    # TODO: Add tests with recycling streams. => Iteration required?
    # TODO: Add tests with multiple ports.
)
class TestSystemConnectivity:
    """Test system connectivity class."""

    def test_connections_matrix(
        self, system: SystemBase, connections, expected_matrix, expected_state
    ):
        """Test for computing the connectivity."""
        system._compute_connectivity_matrix(connections)
        np.testing.assert_almost_equal(system._connectivity, expected_matrix)

    def test_coupling(
        self, system: SystemBase, connections, expected_matrix, expected_state
    ):
        """Test for updating and coupling the system connectivity."""
        y = np.arange(system.n_dof)
        y_dot = 2 * y

        system.y = y
        system.y_dot = y_dot

        system.update_system_connectivity(connections)
        system.couple_unit_operations()
        np.testing.assert_almost_equal(system.y, expected_state)


@pytest.mark.parametrize(
    "system, connections, initial_values, expected_state",
    [
        (
            SystemFixture(),
            [[0, 1, 0, 0, 1e-3], [1, 2, 0, 0, 1e-3]],
            [
                0,
                1,
                0,
                1,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                22,
                23,
            ],
            [
                0,
                1,
                0,
                1,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                22,
                23,
            ],
        )
    ],
)
class TestSystemInitializeInitialValue:
    """Class to test the Initialization of Initial Values."""

    def test_initialize(
        self, system: SystemBase, connections, initial_values, expected_state
    ):
        """Test to check calculation of Initial Values."""
        system.y = initial_values
        system.update_system_connectivity(connections)
        # system.initialize_initial_values(0)


# %% TODO: System Residual


if __name__ == "__main__":
    pytest.main(["test_system.py"])
