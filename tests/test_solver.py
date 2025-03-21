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
    """Solver Fixture Class for testing Solver."""

    def __init__(self, system=None, sections=None, *args, **kwargs):
        """Init of the Solver Fixture with pre set sections."""
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
                            'c_poly': [
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
                            'c_poly': [
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                            ],
                        },
                    },
                },
            ]

        if system is None:
            system = SystemFixture()

        super().__init__(system, sections, *args, **kwargs)


class SystemFixture(FlowSystem):
    """System Fixture Class for Testing purposes."""

    def __init__(self, unit_operations=None, *args, **kwargs):
        """Init System Fixture with set untitoperations when not given."""
        if unit_operations is None:
            unit_operations = [
                InletFixture(),
                DeadEndFiltrationFixture(),
                OutletFixture()
            ]

        super().__init__(unit_operations, *args, **kwargs)
        self.initialize_state()


# %% State Structure
solver_fixture_obj = SolverFixture()
rej_obj = \
    solver_fixture_obj.system.unit_operations['dead_end_filtration'].rejection_model
viscosity_obj = \
    solver_fixture_obj.system.unit_operations['dead_end_filtration'].viscosity_model
@pytest.mark.parametrize(
    "solver, expected",
    [
        (
            solver_fixture_obj,
            {
                'n_dof': 29,
                'unit_solution': {
                    'inlet': {
                        'outlet': {
                            'c': {
                                'values': np.array([[0., 1.], [0., 2.]]),
                                'derivatives': np.array([[0., 2.], [0., 4.]])
                                },
                        }
                    },
                    'dead_end_filtration': {
                        'inlet':{
                            'c': {
                                'values': np.array([[2., 3.], [4., 6.]]),
                                'derivatives': np.array([[4., 6.], [8., 12.]]),
                            },
                            'n': {
                                'values': np.array([[4., 5.], [8., 10.]]),
                                'derivatives': np.array([[8., 10.], [16., 20.]]),
                            },
                        },
                        'permeate': {
                            'c': {
                                'values': np.array([[6., 7.], [12., 14.]]),
                                'derivatives': np.array([[12., 14. ], [24., 28.]]),
                            },
                            'n': {
                                'values': np.array([[8., 9.], [16., 18.]]),
                                'derivatives': np.array([[16., 18.], [32., 36]]),
                            },
                            'V': {
                                'values': np.array([[10.], [20.]]),
                                'derivatives': np.array([[20.], [40.]]),
                            },
                        },
                        'retentate': {
                            'c': {
                                'values': np.array([[11., 12.], [22., 24.]]),
                                'derivatives': np.array([[22., 24.], [44., 48.]]),
                            },
                            'n': {
                                'values': np.array([[13., 14.], [26., 28.]]),
                                'derivatives': np.array([[26., 28.], [52., 56]]),
                            },
                            'V': {
                                'values': np.array([[15.], [30.]]),
                                'derivatives': np.array([[30.], [60.]]),
                            },
                        },
                        'cake': {
                            'c': {
                                'values': np.array([[16., 17.], [32., 34.]]),
                                'derivatives': np.array([[32., 34.], [64., 68.]]),
                            },
                            'n':{
                                'values': np.array([[18., 19.], [36., 38.]]),
                                'derivatives': np.array([[36., 38.], [72., 76.]]),
                            },
                            'V': {
                                'values': np.array([[20.], [40.]]),
                                'derivatives': np.array([[40.], [80.]]),
                            },
                            'pressure': {
                                'values': np.array([[21.], [42.]]),
                                'derivatives': np.array([[42.], [84.]]),
                            },
                        },
                        'permeate_tank': {
                            'c': {
                                'values': np.array([[22., 23.], [44., 46.]]),
                                'derivatives': np.array([[44., 46.], [88., 92.]]),
                            },
                            'n':{
                                'values': np.array([[24., 25.], [48., 50.]]),
                                'derivatives': np.array([[48., 50.], [96., 100.]]),
                            },
                            'V': {
                                'values': np.array([[26.], [52.]]),
                                'derivatives': np.array([[52.], [104.]]),
                            },
                        }
                    },
                    'outlet': {
                        'inlet': {
                            'c': {
                                'values': np.array([[27., 28.], [54., 56.]]),
                                'derivatives': np.array([[54., 56.], [108., 112]]),
                            },
                        },
                    },
                },
                'section_states': {
                    0: {
                        'parameters_start': {
                            'inlet': {
                                'c_poly': np.array([1., 2.]),
                            },
                            'dead_end_filtration': {
                                'rejection_model': rej_obj,
                                'viscosity_model': viscosity_obj,
                                'membrane_area': 1,
                                'membrane_resistance': 1
                            },
                            'outlet': {},
                        },
                        'parameters_end': {
                            'inlet': {
                                'c_poly': np.array([1., 2.]),
                            },
                            'dead_end_filtration': {
                                'rejection_model': rej_obj,
                                'viscosity_model': viscosity_obj,
                                'membrane_area': 1,
                                'membrane_resistance': 1
                            },
                            'outlet': {},
                        },
                    },
                    1: {
                        'parameters_start': {
                            'inlet': {
                                'c_poly': np.array([0., 1.]),
                            },
                            'dead_end_filtration': {
                                'rejection_model': rej_obj,
                                'viscosity_model': viscosity_obj,
                                'membrane_area': 1,
                                'membrane_resistance': 1,
                            },
                            'outlet': {},
                        },
                        'parameters_end': {
                            'inlet': {
                                'c_poly': np.array([2., 1.]),
                            },
                            'dead_end_filtration': {
                                'rejection_model': rej_obj,
                                'viscosity_model': viscosity_obj,
                                'membrane_area': 1,
                                'membrane_resistance': 1,
                            },
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
        ydot = 2 * solution
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

    def test_parameters(self, solver: Solver, expected):
        """Test updating of parameters."""
        for i_sec, section in enumerate(solver.sections):
            solver._update_unit_operation_parameters(
                section['start'], section['end'], section['section_states']
            )

            for unit in solver.system.unit_operations.values():
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
