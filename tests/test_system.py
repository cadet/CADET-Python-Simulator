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
                'n_dof': 16,
                'unit_slices': {
                    'inlet': slice(0, 3, None),
                    'dead_end_filtration': slice(3, 13, None),
                    'outlet': slice(13, 16, None),
                },
                'unit_solution': {
                    'inlet': {
                        'c_out': np.array([[0., 1.], [0., 2.]]),
                        'c_out_dot': np.array([[ 0.,  2.], [0., 4.]]),
                        'viscosity_out': np.array([[ 2.], [4.]]),
                        'viscosity_out_dot': np.array([[4.], [8.]]),
                    },
                    'dead_end_filtration': {
                        'c_in': np.array([[ 3.,  4.], [6., 8.]]),
                        'c_in_dot': np.array([[ 6.,  8.], [12., 16.]]),
                        'viscosity_in': np.array([[ 5.], [10.]]),
                        'viscosity_in_dot': np.array([[10.], [20.]]),
                        'Vp': np.array([[ 6.], [12.]]),
                        'Vp_dot': np.array([[12.], [24.]]),
                        'Rc': np.array([[7.], [14.]]),
                        'Rc_dot': np.array([[14.], [28.]]),
                        'mc': np.array([[8., 9.], [16., 18.]]),
                        'mc_dot': np.array([[16., 18.], [32., 36.]]),
                        'c_out': np.array([[10., 11.], [20., 22.]]),
                        'c_out_dot': np.array([[20., 22.], [40., 44.]]),
                        'viscosity_out': np.array([[12.], [24.]]),
                        'viscosity_out_dot': np.array([[24.], [48.]]),
                    },
                    'outlet': {
                        'c_in': np.array([[13., 14.], [26., 28.]]),
                        'c_in_dot': np.array([[26., 28.], [52., 56.]]),
                        'viscosity_in': np.array([[15.], [30.]]),
                        'viscosity_in_dot': np.array([[30.], [60.]]),
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

    def test_system_structure(self, system, expected):
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
            [0, 1, 2, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11],
        ),
    ]
    # TODO: Add tests with split/combined streams.
    # TODO: Add tests with recycling streams. => Iteration required?
    # TODO: Add tests with multiple ports.
)
class TestSystemConnectivity():
    def test_connections_matrix(
            self,
            system,
            connections,
            expected_matrix,
            expected_state
            ):
        matrix = system._compute_connectivity_matrix(connections)
        np.testing.assert_almost_equal(matrix, expected_matrix)

    def test_coupling(
            self,
            system: SystemBase,
            connections,
            expected_matrix,
            expected_state
            ):

        y = np.arange(system.n_dof)
        y_dot = 2 * y

        system.y = y
        system.y_dot = y_dot

        system.update_system_connectivity(connections)
        system.couple_unit_operations()
        np.testing.assert_almost_equal(system.y, expected_state)

# %% TODO: System Residual


if __name__ == "__main__":
    pytest.main(["test_system.py"])
