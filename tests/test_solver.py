import numpy as np
import pytest

from test_unit_operation import (
    InletFixture, OutletFixture, CstrFixture, DeadEndFiltrationFixture
)
from CADETPythonSimulator.system import SystemBase
from CADETPythonSimulator.solver import Solver


# %% System Fixtures
class SolverFixture(Solver):
    def __init__(self, system=None, sections=None, *args, **kwargs):
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

        if system is None:
            system=SystemFixture()

        super().__init__(system, sections, *args, **kwargs)

class SystemFixture(SystemBase):
    def __init__(self, unit_operations=None, *args, **kwargs):
        if unit_operations is None:
            unit_operations = [
                InletFixture(),
                DeadEndFiltrationFixture(),
                OutletFixture()
            ]

        super().__init__(unit_operations, *args, **kwargs)


# %% State Structure

@pytest.mark.parametrize(
    "solver, expected",
    [
        (
            SolverFixture(),
            {
                'n_dof': 16,
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
class TestSolver():

    def test_solution_recorder(self, solver: Solver, expected):
        """Test the solution recorder."""
        solver.initialize_solution_recorder()

        solution = np.arange(solver._system.n_dof)
        solver.write_solution(solution, 2 * solution)
        solver.write_solution(2 * solution, 4 * solution)

        unit_solutions = solver.unit_solutions
        for unit, solutions in unit_solutions.items():
            for state, values in solutions.items():
                np.testing.assert_almost_equal(
                    values, expected['unit_solution'][str(unit)][state]
                )

    def test_section_states(self, solver, expected):
        for i_sec, section in enumerate(solver.sections):
            solver._update_section_states(
                section['start'], section['end'], section['section_states']
            )

            for unit in solver.system.units:
                expected_values = expected['section_states'][i_sec]

                values_start = unit.get_parameter_values_at_time(section['start'])
                np.testing.assert_equal(
                    values_start, expected_values['parameters_start'][str(unit)]
                )

                values_end = unit.get_parameter_values_at_time(section['end'])
                np.testing.assert_equal(
                    values_end, expected_values['parameters_end'][str(unit)]
                )


# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_solver.py"])
