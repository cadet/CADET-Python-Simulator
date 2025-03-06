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
                'n_dof': 22,
                'unit_slices': {
                    'inlet': slice(0, 2, None),
                    'dead_end_filtration': slice(2, 11, None),
                    'outlet': slice(11, 13, None),
                },
                'unit_solution': {
                    'inlet': {
                        'c_out': np.array([[0., 1.], [0., 2.]]),
                        'c_out_dot': np.array([[ 0.,  2.], [0., 4.]]),
                    },
                    'dead_end_filtration': {
                        'c_in': np.array([[ 2.,  3.], [4., 6.]]),
                        'c_in_dot': np.array([[ 4.,  6.], [8., 12.]]),
                        'Vp': np.array([[ 4.], [8.]]),
                        'Vp_dot': np.array([[8.], [16.]]),
                        'Rc': np.array([[5.], [10.]]),
                        'Rc_dot': np.array([[10.], [20.]]),
                        'mc': np.array([[6., 7.], [12., 14.]]),
                        'mc_dot': np.array([[12., 14.], [24., 28.]]),
                        'c_out': np.array([[8., 9.], [16., 18.]]),
                        'c_out_dot': np.array([[16., 18.], [32., 36.]]),
                    },
                    'outlet': {
                        'c_in': np.array([[10., 11.], [20., 22.]]),
                        'c_in_dot': np.array([[20., 22.], [40., 44.]]),
                    },
                },
                'section_states': {
                    0: {
                        'parameters_start': {
                            'inlet': {
                                'c': np.array([1., 2.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                        'parameters_end': {
                            'inlet': {
                                'c': np.array([1., 2.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                    },
                    1: {
                        'parameters_start': {
                            'inlet': {
                                'c': np.array([0., 1.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                        'parameters_end': {
                            'inlet': {
                                'c': np.array([2., 1.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                    },
                },
            },
        ),
    ]
)
class TestSystem():
    """Test Class for system."""

    def test_system_structure(self, system, expected):
        """Test system structure."""
        assert system.n_dof == expected['n_dof']


# %% System Connectivity

@pytest.mark.parametrize(
    "system, connections, expected_matrix, expected_state",
    [
        (
            SystemFixture(),
            [[0, 1, 0, 0, 1e-3], [1, 2, 0, 0, 1e-3]],
            [
                [0.001, 0.   ],
                [0.   , 0.001]
            ],
            [ 0,  1,  0,  1,  4,  5,  6,  7,  8,  9, 10,\
             11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16],
        ),
    ]
    # TODO: Add tests with split/combined streams.
    # TODO: Add tests with recycling streams. => Iteration required?
    # TODO: Add tests with multiple ports.
)
class TestSystemConnectivity():
    """Test system connectivity class."""

    def test_connections_matrix(
            self,
            system: SystemBase,
            connections,
            expected_matrix,
            expected_state
            ):
        """Test for computing the connectivity."""
        system._compute_connectivity_matrix(connections)
        np.testing.assert_almost_equal(system._connectivity, expected_matrix)

    def test_coupling(
            self,
            system: SystemBase,
            connections,
            expected_matrix,
            expected_state
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
            [ 0,  1,  0,  1,  4,  5,  6,  7,  8,  9, 10,\
             11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16],
            [ 0,  1,  0,  1,  4,  5,  6,  7,  8,  9, 10,\
             11, 12, 13, 14, 15, 16, 17, 18, 19, 15, 16],
        ),
    ]
)
class TestSystemInitializeInitialValue():
    """Class to test the Initialization of Initial Values."""

    def test_initialize(
            self,
            system: SystemBase,
            connections,
            initial_values,
            expected_state
            ):
        """Test to check calculation of Initial Values."""
        system.y = initial_values
        system.update_system_connectivity(connections)
        #system.initialize_initial_values(0)


# %% TODO: System Residual


if __name__ == "__main__":
    pytest.main(["test_system.py"])
