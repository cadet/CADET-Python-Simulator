import numpy as np
import pytest
import xarray as xr
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.distribution_base import (
    ConstantConcentrationDistribution,
    ConstantVolumeDistribution,
)
from CADETPythonSimulator.rejection import StepCutOff
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.unit_operation import DeadEndFiltration, DistributionInlet
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity

# ── fixture ───────────────────────────────────────────────────────────────────
# Component system: 3 components with MW [1e6, 8e4, 18].
# StepCutOff at 1.6e5: component 0 (MW=1e6) is retained, 1 and 2 pass through.


@pytest.fixture(scope="module")
def filter_solution():
    component_system = CPSComponentSystem(
        name="test_comp",
        components=3,
        densities=[1100, 1100, 1000],
        molecular_weights=[1e6, 8e4, 18],
        viscosities=[np.nan, np.nan, 0.001],
        specific_cake_resistances=[1e6, 1e6, 0],
    )
    concentration_distribution = ConstantVolumeDistribution(
        component_system=component_system, c=[1e-10, 1e-8]
    )
    inlet = DistributionInlet(component_system=component_system, name="inlet")
    inlet.distribution_function = concentration_distribution

    filter_obj = DeadEndFiltration(
        component_system=component_system,
        name="deadendfilter",
        rejection_model=StepCutOff(cutoff_weight=2 * 8e4),
        viscosity_model=LogarithmicMixingViscosity(),
        membrane_area=1,
        membrane_resistance=1,
    )

    system = FlowSystem(unit_operations=[inlet, filter_obj])
    system.initialize_state()

    condist = ConstantConcentrationDistribution(
        component_system=component_system, c=[0, 0]
    )
    system.states["deadendfilter"]["permeate_tank"]["V"] = 1e-9
    system.states["deadendfilter"]["permeate_tank"]["c"] = condist.get_distribution(
        0, 0
    )

    solver = Solver(
        system,
        [{"start": 0, "end": 11, "connections": [("inlet", "deadendfilter", 1)]}],
    )
    solver.solve()
    return solver


# ── output structure ──────────────────────────────────────────────────────────


class TestFilterOutputStructure:
    def test_time_has_multiple_points(self, filter_solution):
        assert len(filter_solution.time_solutions) > 1

    def test_time_is_sorted(self, filter_solution):
        t = filter_solution.time_solutions
        assert np.all(t[1:] >= t[:-1])

    def test_unit_solutions_are_datarrays(self, filter_solution):
        sol = filter_solution.unit_solutions
        for states in sol.values():
            for entries in states.values():
                for da in entries.values():
                    assert isinstance(da, xr.DataArray)

    def test_time_dim_present(self, filter_solution):
        sol = filter_solution.unit_solutions
        for states in sol.values():
            for entries in states.values():
                for da in entries.values():
                    assert "time" in da.dims

    def test_multicomp_entries_have_component_dim(self, filter_solution):
        da = filter_solution.unit_solutions["deadendfilter"]["cake"]["n"]
        assert "component" in da.dims
        assert da.sizes["component"] == 3

    def test_scalar_entries_have_no_component_dim(self, filter_solution):
        da = filter_solution.unit_solutions["deadendfilter"]["cake"]["V"]
        assert "component" not in da.dims

    def test_time_coordinate_matches_time_solutions(self, filter_solution):
        da = filter_solution.unit_solutions["deadendfilter"]["cake"]["n"]
        np.testing.assert_array_equal(
            da.coords["time"].values, filter_solution.time_solutions
        )

    def test_no_nans(self, filter_solution):
        sol = filter_solution.unit_solutions
        for unit, states in sol.items():
            for state, entries in states.items():
                for entry, da in entries.items():
                    assert not np.any(np.isnan(da.values)), (
                        f"NaN in {unit}.{state}.{entry}"
                    )


# ── physics ───────────────────────────────────────────────────────────────────


class TestFilterPhysics:
    def test_retained_component_accumulates_in_cake(self, filter_solution):
        # Component 0 (MW=1e6) is above the cutoff and should be retained.
        n_cake = filter_solution.unit_solutions["deadendfilter"]["cake"]["n"]
        assert n_cake.values[-1, 0] > n_cake.values[0, 0]

    def test_passing_components_do_not_accumulate_in_cake(self, filter_solution):
        # Components 1 and 2 pass through; cake n for them stays near zero.
        n_cake = filter_solution.unit_solutions["deadendfilter"]["cake"]["n"]
        assert np.allclose(n_cake.values[:, 1], 0, atol=1e-20)
        assert np.allclose(n_cake.values[:, 2], 0, atol=1e-20)

    def test_permeate_tank_fills(self, filter_solution):
        V_tank = filter_solution.unit_solutions["deadendfilter"]["permeate_tank"]["V"]
        assert V_tank.values[-1] > V_tank.values[0]

    def test_pressure_positive(self, filter_solution):
        pressure = filter_solution.unit_solutions["deadendfilter"]["cake"]["pressure"]
        assert np.all(pressure.values > 0)

    def test_pressure_increases_with_cake(self, filter_solution):
        pressure = filter_solution.unit_solutions["deadendfilter"]["cake"]["pressure"]
        assert np.all(pressure.values[1:] >= pressure.values[:-1])


# ── auxiliary state ───────────────────────────────────────────────────────────


class TestAuxiliaryState:
    def test_cake_height_present(self, filter_solution):
        assert "cake_aux" in filter_solution.unit_solutions["deadendfilter"]
        assert (
            "cake_height" in filter_solution.unit_solutions["deadendfilter"]["cake_aux"]
        )

    def test_cake_height_is_dataarray(self, filter_solution):
        da = filter_solution.unit_solutions["deadendfilter"]["cake_aux"]["cake_height"]
        assert isinstance(da, xr.DataArray)

    def test_cake_height_increases(self, filter_solution):
        # Cake grows monotonically as retained material accumulates.
        h = filter_solution.unit_solutions["deadendfilter"]["cake_aux"]["cake_height"]
        assert np.all(h.values[1:] >= h.values[:-1])

    def test_cake_height_no_nans(self, filter_solution):
        h = filter_solution.unit_solutions["deadendfilter"]["cake_aux"]["cake_height"]
        assert not np.any(np.isnan(h.values))

    def test_pressure_increases_with_cake(self, filter_solution):
        # As cake resistance grows, pressure should increase monotonically.
        pressure = filter_solution.unit_solutions["deadendfilter"]["cake"]["pressure"]
        assert np.all(pressure.values[1:] >= pressure.values[:-1])
