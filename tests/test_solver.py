import numpy as np
import pytest
from addict import Dict
from typing import NoReturn, Optional

from test_unit_operation import (
    InletFixture,
    OutletFixture,
    CstrFixture,
    DeadEndFiltrationFixture,
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
                    "start": 0,
                    "end": 10,
                    "connections": [[0, 1, 1e-3], [1, 2, 1e-3]],
                    "section_states": {
                        "inlet": {"c_poly": [[1, 0, 0, 0], [2, 0, 0, 0]]}
                    },
                },
                {
                    "start": 10,
                    "end": 12,
                    "connections": [[0, 1, 1e-3], [1, 2, 0.5e-3]],
                    "section_states": {
                        "inlet": {"c_poly": [[0, 1, 0, 0], [1, 0, 0, 0]]}
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
                OutletFixture(),
            ]

        super().__init__(unit_operations, *args, **kwargs)
        self.initialize_state()


# %% State Structure
solver_fixture_obj = SolverFixture()
rej_obj = solver_fixture_obj.system.unit_operations[
    "dead_end_filtration"
].rejection_model
viscosity_obj = solver_fixture_obj.system.unit_operations[
    "dead_end_filtration"
].viscosity_model


@pytest.mark.parametrize(
    "solver, expected",
    [
        (
            solver_fixture_obj,
            {
                "n_dof": 29,
                "unit_solution": {
                    "inlet": {
                        "outlet": {
                            "c": {
                                "values": np.array([[0.0, 1.0], [0.0, 2.0]]),
                                "derivatives": np.array([[0.0, 2.0], [0.0, 4.0]]),
                            }
                        }
                    },
                    "dead_end_filtration": {
                        "inlet": {
                            "c": {
                                "values": np.array([[2.0, 3.0], [4.0, 6.0]]),
                                "derivatives": np.array([[4.0, 6.0], [8.0, 12.0]]),
                            },
                            "n": {
                                "values": np.array([[4.0, 5.0], [8.0, 10.0]]),
                                "derivatives": np.array([[8.0, 10.0], [16.0, 20.0]]),
                            },
                        },
                        "permeate": {
                            "c": {
                                "values": np.array([[6.0, 7.0], [12.0, 14.0]]),
                                "derivatives": np.array([[12.0, 14.0], [24.0, 28.0]]),
                            },
                            "n": {
                                "values": np.array([[8.0, 9.0], [16.0, 18.0]]),
                                "derivatives": np.array([[16.0, 18.0], [32.0, 36]]),
                            },
                            "V": {
                                "values": np.array([[10.0], [20.0]]),
                                "derivatives": np.array([[20.0], [40.0]]),
                            },
                        },
                        "retentate": {
                            "c": {
                                "values": np.array([[11.0, 12.0], [22.0, 24.0]]),
                                "derivatives": np.array([[22.0, 24.0], [44.0, 48.0]]),
                            },
                            "n": {
                                "values": np.array([[13.0, 14.0], [26.0, 28.0]]),
                                "derivatives": np.array([[26.0, 28.0], [52.0, 56]]),
                            },
                            "V": {
                                "values": np.array([[15.0], [30.0]]),
                                "derivatives": np.array([[30.0], [60.0]]),
                            },
                        },
                        "cake": {
                            "c": {
                                "values": np.array([[16.0, 17.0], [32.0, 34.0]]),
                                "derivatives": np.array([[32.0, 34.0], [64.0, 68.0]]),
                            },
                            "n": {
                                "values": np.array([[18.0, 19.0], [36.0, 38.0]]),
                                "derivatives": np.array([[36.0, 38.0], [72.0, 76.0]]),
                            },
                            "V": {
                                "values": np.array([[20.0], [40.0]]),
                                "derivatives": np.array([[40.0], [80.0]]),
                            },
                            "pressure": {
                                "values": np.array([[21.0], [42.0]]),
                                "derivatives": np.array([[42.0], [84.0]]),
                            },
                        },
                        "permeate_tank": {
                            "c": {
                                "values": np.array([[22.0, 23.0], [44.0, 46.0]]),
                                "derivatives": np.array([[44.0, 46.0], [88.0, 92.0]]),
                            },
                            "n": {
                                "values": np.array([[24.0, 25.0], [48.0, 50.0]]),
                                "derivatives": np.array([[48.0, 50.0], [96.0, 100.0]]),
                            },
                            "V": {
                                "values": np.array([[26.0], [52.0]]),
                                "derivatives": np.array([[52.0], [104.0]]),
                            },
                        },
                    },
                    "outlet": {
                        "inlet": {
                            "c": {
                                "values": np.array([[27.0, 28.0], [54.0, 56.0]]),
                                "derivatives": np.array([[54.0, 56.0], [108.0, 112]]),
                            }
                        }
                    },
                },
                "section_states": {
                    0: {
                        "parameters_start": {
                            "inlet": {"c_poly": np.array([1.0, 2.0])},
                            "dead_end_filtration": {
                                "rejection_model": rej_obj,
                                "viscosity_model": viscosity_obj,
                                "membrane_area": 1,
                                "membrane_resistance": 1,
                            },
                            "outlet": {},
                        },
                        "parameters_end": {
                            "inlet": {"c_poly": np.array([1.0, 2.0])},
                            "dead_end_filtration": {
                                "rejection_model": rej_obj,
                                "viscosity_model": viscosity_obj,
                                "membrane_area": 1,
                                "membrane_resistance": 1,
                            },
                            "outlet": {},
                        },
                    },
                    1: {
                        "parameters_start": {
                            "inlet": {"c_poly": np.array([0.0, 1.0])},
                            "dead_end_filtration": {
                                "rejection_model": rej_obj,
                                "viscosity_model": viscosity_obj,
                                "membrane_area": 1,
                                "membrane_resistance": 1,
                            },
                            "outlet": {},
                        },
                        "parameters_end": {
                            "inlet": {"c_poly": np.array([2.0, 1.0])},
                            "dead_end_filtration": {
                                "rejection_model": rej_obj,
                                "viscosity_model": viscosity_obj,
                                "membrane_area": 1,
                                "membrane_resistance": 1,
                            },
                            "outlet": {},
                        },
                    },
                },
            },
        )
    ],
)
class TestSolver:
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
                            expected["unit_solution"][unit][state][d_of][sol_type],
                        )

    def test_parameters(self, solver: Solver, expected):
        """Test updating of parameters."""
        for i_sec, section in enumerate(solver.sections):
            solver._update_unit_operation_parameters(
                section["start"], section["end"], section["section_states"]
            )

            for unit in solver.system.unit_operations.values():
                expected_values = expected["section_states"][i_sec]

                values_start = unit.get_parameter_values_at_time(section["start"])

                np.testing.assert_equal(
                    values_start, expected_values["parameters_start"][str(unit)]
                )

                values_end = unit.get_parameter_values_at_time(section["end"])

                np.testing.assert_equal(
                    values_end, expected_values["parameters_end"][str(unit)]
                )


# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_solver.py"])
