import numpy as np
import pytest


from CADETPythonSimulator.distribution_base import (
    ConstantVolumeDistribution,
    ConstantConcentrationDistribution,
)

from CADETPythonSimulator.unit_operation import (
    DistributionInlet,
    Outlet,
    DeadEndFiltration,
)
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.rejection import StepCutOff
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity


def setup_filter_sim():
    """Set up function for filter."""
    component_system = CPSComponentSystem(
        name="test_comp",
        components=3,
        densities=[1100, 1100, 1000],
        molecular_weights=[1e6, 8e4, 18],
        viscosities=[np.nan, np.nan, 0.001],
        specific_cake_resistances=[1e6, 1e6, 0],
    )

    # Concidering c3 is unknown.
    concentration_distribution = ConstantVolumeDistribution(
        component_system=component_system, c=[1e-10, 1e-8]
    )
    inlet = DistributionInlet(component_system=component_system, name="inlet")
    inlet.distribution_function = concentration_distribution

    rejectionmodell = StepCutOff(cutoff_weight=2 * 8e4)
    viscositymodell = LogarithmicMixingViscosity()
    filter_obj = DeadEndFiltration(
        component_system=component_system,
        name="deadendfilter",
        rejection_model=rejectionmodell,
        viscosity_model=viscositymodell,
        membrane_area=1,
        membrane_resistance=1,
    )

    # outlet = Outlet(component_system=component_system, name="outlet")

    unit_operation_list = [inlet, filter_obj]

    system = FlowSystem(unit_operations=unit_operation_list)
    section = [{"start": 0, "end": 11, "connections": [[0, 1, 0, 0, 1]]}]

    system.initialize_state()

    condist = ConstantConcentrationDistribution(
        component_system=component_system, c=[0, 0]
    )
    c_init = condist.get_distribution(0, 0)

    """
    Setting startingvolume and accoding Concentrations are necessary.
    Permeate Tank must not be empty.
    """
    system.states["deadendfilter"]["permeate_tank"]["V"] = 1e-9
    system.states["deadendfilter"]["permeate_tank"]["c"] = c_init

    solver = Solver(system, section)
    return solver


@pytest.mark.parametrize("setup_func, expected", [(setup_filter_sim, 3)])
class Testsimulation:
    """Class tries to run an actual Simulation."""

    def test_solver_convergence(self, setup_func, expected):
        """Test wether the simulation converges."""
        solver = setup_func()
        solver.solve()
        assert len(solver.time_solutions) > 1
