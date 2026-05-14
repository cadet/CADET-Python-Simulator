import numpy as np
import pytest
import xarray as xr

from CADETPythonSimulator.solver import Solver

from conftest import SolveDummy, SystemFixture, SolverFixture


# %% State Structure
solver_fixture_obj = SolverFixture()
rej_obj = solver_fixture_obj.system.unit_operations[
    "dead_end_filtration"
].rejection_model
viscosity_obj = solver_fixture_obj.system.unit_operations[
    "dead_end_filtration"
].viscosity_model

# Expected arrays: shape (T, n_comp) for multi-component entries, (T,) for scalars.
# Two time steps: t=0 with y=arange(29), t=1 with y=2*arange(29).
_expected_solutions = {
    "inlet": {
        "outlet": {"c": np.array([[0.0, 1.0], [0.0, 2.0]])},
    },
    "dead_end_filtration": {
        "inlet": {
            "c": np.array([[2.0, 3.0], [4.0, 6.0]]),
            "n": np.array([[4.0, 5.0], [8.0, 10.0]]),
        },
        "permeate": {
            "c": np.array([[6.0, 7.0], [12.0, 14.0]]),
            "n": np.array([[8.0, 9.0], [16.0, 18.0]]),
            "V": np.array([10.0, 20.0]),
        },
        "retentate": {
            "c": np.array([[11.0, 12.0], [22.0, 24.0]]),
            "n": np.array([[13.0, 14.0], [26.0, 28.0]]),
            "V": np.array([15.0, 30.0]),
        },
        "cake": {
            "c": np.array([[16.0, 17.0], [32.0, 34.0]]),
            "n": np.array([[18.0, 19.0], [36.0, 38.0]]),
            "V": np.array([20.0, 40.0]),
            "pressure": np.array([21.0, 42.0]),
        },
        "permeate_tank": {
            "c": np.array([[22.0, 23.0], [44.0, 46.0]]),
            "n": np.array([[24.0, 25.0], [48.0, 50.0]]),
            "V": np.array([26.0, 52.0]),
        },
    },
    "outlet": {
        "inlet": {"c": np.array([[27.0, 28.0], [54.0, 56.0]])},
    },
}

# Derivatives = 2 * y at each step (ydot passed as 2*y and 4*y respectively).
_expected_derivatives = {
    "inlet": {
        "outlet": {"c": np.array([[0.0, 2.0], [0.0, 4.0]])},
    },
    "dead_end_filtration": {
        "inlet": {
            "c": np.array([[4.0, 6.0], [8.0, 12.0]]),
            "n": np.array([[8.0, 10.0], [16.0, 20.0]]),
        },
        "permeate": {
            "c": np.array([[12.0, 14.0], [24.0, 28.0]]),
            "n": np.array([[16.0, 18.0], [32.0, 36.0]]),
            "V": np.array([20.0, 40.0]),
        },
        "retentate": {
            "c": np.array([[22.0, 24.0], [44.0, 48.0]]),
            "n": np.array([[26.0, 28.0], [52.0, 56.0]]),
            "V": np.array([30.0, 60.0]),
        },
        "cake": {
            "c": np.array([[32.0, 34.0], [64.0, 68.0]]),
            "n": np.array([[36.0, 38.0], [72.0, 76.0]]),
            "V": np.array([40.0, 80.0]),
            "pressure": np.array([42.0, 84.0]),
        },
        "permeate_tank": {
            "c": np.array([[44.0, 46.0], [88.0, 92.0]]),
            "n": np.array([[48.0, 50.0], [96.0, 100.0]]),
            "V": np.array([52.0, 104.0]),
        },
    },
    "outlet": {
        "inlet": {"c": np.array([[54.0, 56.0], [108.0, 112.0]])},
    },
}

_section_states = {
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
}


@pytest.mark.parametrize("solver", [solver_fixture_obj])
class TestSolver:
    """Testing the Solver Object."""

    def _write_two_steps(self, solver: Solver):
        solver.initialize_solution_recorder()
        n = solver._system.n_dof
        base = np.arange(n, dtype=float).reshape(1, n)
        solver.write_solution([0], base, 2 * base)
        solver.write_solution([1], 2 * base, 4 * base)

    def test_solution_recorder_values(self, solver: Solver):
        """unit_solutions entries are DataArrays with correct values."""
        self._write_two_steps(solver)
        sol = solver.unit_solutions
        for unit, states in _expected_solutions.items():
            for state, entries in states.items():
                for entry, expected in entries.items():
                    da = sol[unit][state][entry]
                    assert isinstance(da, xr.DataArray)
                    assert "time" in da.dims
                    np.testing.assert_almost_equal(da.values, expected)

    def test_solution_recorder_derivatives(self, solver: Solver):
        """unit_derivatives entries are DataArrays with correct values."""
        self._write_two_steps(solver)
        deriv = solver.unit_derivatives
        for unit, states in _expected_derivatives.items():
            for state, entries in states.items():
                for entry, expected in entries.items():
                    da = deriv[unit][state][entry]
                    assert isinstance(da, xr.DataArray)
                    assert "time" in da.dims
                    np.testing.assert_almost_equal(da.values, expected)

    def test_time_solutions(self, solver: Solver):
        """time_solutions reflects the written time points."""
        self._write_two_steps(solver)
        np.testing.assert_array_equal(solver.time_solutions, [0.0, 1.0])

    def test_parameters(self, solver: Solver):
        """Test updating of parameters."""
        for i_sec, section in enumerate(solver.sections):
            solver._update_unit_operation_parameters(
                section["start"], section["end"], section["section_states"]
            )

            for unit in solver.system.unit_operations.values():
                expected_values = _section_states[i_sec]

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
