import numpy as np
import pytest
from addict import Dict
from typing import NoReturn, Optional

from test_unit_operation import (
    InletFixture, OutletFixture, CstrFixture, DeadEndFiltrationFixture
)
from CADETProcess.dataStructure import Structure
from CADETPythonSimulator.system import SystemBase, FlowSystem
from CADETPythonSimulator.solver import Solver


class SolveDummy(Structure):
    """Dummy class to Simulate a Solution Object of DAE."""

    def __init__(self, test_dict=None):
        """Construct a Solution Dummy."""
        self._test_dict: Optional[Dict] = test_dict

    @property
    def values(self):
        """Values Property to imitate."""
        return self._test_dict

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
            system.initialize()

        super().__init__(system, sections, *args, **kwargs)

class SystemFixture(FlowSystem):
    def __init__(self, unit_operations=None, *args, **kwargs):
        if unit_operations is None:
            unit_operations = [
                InletFixture(),
                DeadEndFiltrationFixture(),
                OutletFixture()
            ]

        super().__init__(unit_operations, *args, **kwargs)
        self.initialize()


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
                        'outlet':{
                            'c': {
                                'values': np.array([[0., 1.], [0., 2.]]),
                                'derivatives': np.array([[ 0.,  2.], [0., 4.]])
                                },
                            'viscosity': {
                                'values': np.array([[ 2.], [4.]]),
                                'derivatives': np.array([[4.], [8.]]),
                            }
                        }
                    },
                    'dead_end_filtration': {
                        'cake': {
                            'c': {
                                'values': np.array([[ 3.,  4.], [6., 8.]]),
                                'derivatives': np.array([[ 6.,  8.], [12., 16.]]),
                            },
                            'viscosity': {
                                'values': np.array([[ 5.], [10.]]),
                                'derivatives': np.array([[10.], [20.]]),

                            },
                            'pressure': {
                                'values': np.array([[ 6.], [12.]]),
                                'derivatives': np.array([[12.], [24.]]),
                            },
                            'cakevolume':{
                                'values': np.array([[7.], [14.]]),
                                'derivatives': np.array([[14.], [28.]]),
                            },
                            'permeate': {
                                'values': np.array([[8.], [16.]]),
                                'derivatives': np.array([[16.], [32.]]),
                            }
                        },
                        'permeate': {
                            'c': {
                                'values': np.array([[9., 10.], [18., 20.]]),
                                'derivatives': np.array([[18., 20.], [36., 40.]]),
                            },
                            'viscosity': {
                                'values': np.array([[11.], [22.]]),
                                'derivatives': np.array([[22.], [44.]]),
                            },
                            'Volume': {
                                'values': np.array([[12.], [24.]]),
                                'derivatives': np.array([[24.], [48.]]),
                            }
                        }
                    },
                    'outlet': {
                        'inlet': {
                            'c': {
                                'values': np.array([[13., 14.], [26., 28.]]),
                                'derivatives': np.array([[26., 28.], [52., 56.]]),
                            },
                            'viscosity': {
                                'values': np.array([[15.], [30.]]),
                                'derivatives': np.array([[30.], [60.]]),
                            }
                        },
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
    """Testing the Solver Object."""

    def test_solution_recorder(self, solver: Solver, expected):
        """Test the solution recorder."""
        solver.initialize_solution_recorder()

        solution = np.arange(solver._system.n_dof).reshape((-1, solver._system.n_dof))


        y = solution
        ydot = 2* solution
        t = [0]

        solver.write_solution(t, y, ydot)

        y = 2 * solution
        ydot = 4 * solution
        t = [1]

        solver.write_solution(t, y, ydot)

        unit_solutions = solver.unit_solutions
        for unit, solutions in unit_solutions.items():
            for state, struct in solutions.items():
                for d_of, sol in struct.items():
                    for sol_type, sol_array in sol.items():
                        np.testing.assert_almost_equal(
                            sol_array,
                            expected['unit_solution'][unit][state][d_of][sol_type]
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
