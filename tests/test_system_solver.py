import numpy as np
import pytest

from test_unit_operation import (
    InletFixture, OutletFixture, CstrFixture, DeadEndFiltrationFixture
)
from CADETPythonSimulator.system_solver import SystemSolver


# %% System Fixtures

class SystemSolverFixture(SystemSolver):
    def __init__(self, unit_operations=None, sections=None, *args, **kwargs):
        if unit_operations is None:
            unit_operations = [
                InletFixture(),
                DeadEndFiltrationFixture(),
                OutletFixture()
            ]

        if sections is None:
            sections = [
                {
                    'start': 0,
                    'end': 10,
                    'connections': [
                        [0, 1, 1e-3],
                        [1, 2, 1e-3],
                    ],
                    'section_states': {
                        'inlet': {
                            'c': [
                                [1, 0, 0, 0],
                                [2, 0, 0, 0],
                            ],
                        },
                    },
                },
                {
                    'start': 10,
                    'end': 12,
                    'connections': [
                        [0, 1, 1e-3],
                        [1, 2, 0.5e-3],
                    ],
                    'section_states': {
                        'inlet': {
                            'c': [
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                            ],
                        },
                    },
                },
            ]

        super().__init__(unit_operations, sections, *args, **kwargs)


# %% State Structure

@pytest.mark.parametrize(
    "system_solver, expected",
    [
        (
            SystemSolverFixture(),
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
class TestSystemSolverStateStructure():

    def test_state_structure(self, system_solver, expected):
        """Test the system state."""
        assert system_solver.n_dof == expected['n_dof']
        # Transform keys to unit objects
        expected_slices = {
            system_solver.unit_operations_dict[unit]: value
            for unit, value in expected['unit_slices'].items()
        }
        np.testing.assert_equal(system_solver.unit_slices, expected_slices)

    def test_solution_recorder(self, system_solver, expected):
        """Test the solution recorder."""
        system_solver.initialize_solution_recorder()

        solution = np.arange(system_solver.n_dof)
        system_solver.write_solution(solution, 2 * solution)
        system_solver.write_solution(2 * solution, 4 * solution)

        unit_solutions = system_solver.unit_solutions
        for unit, solutions in unit_solutions.items():
            for state, values in solutions.items():
                np.testing.assert_almost_equal(
                    values, expected['unit_solution'][str(unit)][state]
                )

    def test_section_states(self, system_solver, expected):
        for i_sec, section in enumerate(system_solver.sections):
            system_solver._update_section_states(
                section['start'], section['end'], section['section_states']
            )

            for unit in system_solver.units:
                expected_values = expected['section_states'][i_sec]

                values_start = unit.get_parameter_values_at_time(section['start'])
                np.testing.assert_equal(
                    values_start, expected_values['parameters_start'][str(unit)]
                )

                values_end = unit.get_parameter_values_at_time(section['end'])
                np.testing.assert_equal(
                    values_end, expected_values['parameters_end'][str(unit)]
                )


# %% System Connectivity

@pytest.mark.parametrize(
    "system_solver, connections, expected_matrix, expected_state",
    [
        (
            SystemSolverFixture(),
            [[0, 1, 0, 0, 1e-3], [1, 2, 0, 0, 1e-3]],
            [
                [0.001, 0.   ],
                [0.   , 0.001]
            ],
            [0, 1, 2, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12],
        ),
    ]
    # TODO: Add tests with split/combined streams.
    # TODO: Add tests with recycling streams. => Iteration required?
    # TODO: Add tests with multiple ports.
)
class TestSystemConnectivity():
    def test_connections_matrix(
            self,
            system_solver,
            connections,
            expected_matrix,
            expected_state
            ):
        matrix = system_solver._compute_connectivity_matrix(connections)
        np.testing.assert_almost_equal(matrix, expected_matrix)

    def test_coupling(
            self,
            system_solver,
            connections,
            expected_matrix,
            expected_state
            ):

        y = np.arange(system_solver.n_dof)
        y_dot = 2 * y

        system_solver.couple_unit_operations(connections, y, y_dot)
        np.testing.assert_almost_equal(y, expected_state)

# %% TODO: System Residual



# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_system_solver.py"])
