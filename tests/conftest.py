from typing import NoReturn, Optional

import numpy as np
from addict import Dict
from CADETProcess.dataStructure import Structure
from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.rejection import StepCutOff
from CADETPythonSimulator.solver import Solver
from CADETPythonSimulator.system import FlowSystem
from CADETPythonSimulator.unit_operation import (
    _2DGRM,
    CrossFlowFiltration,
    Cstr,
    DeadEndFiltration,
    Inlet,
    Outlet,
    UnitOperationBase,
)
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity


class TwoComponentFixture(CPSComponentSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_component(
            "A", molecular_weight=1e3, density=1e3, molecular_volume=1, viscosity=1
        )
        self.add_component(
            "B", molecular_weight=10e3, density=1e3, molecular_volume=1, viscosity=1
        )


_two_component_system = TwoComponentFixture()


class UnitOperationFixture(UnitOperationBase):
    def __init__(self, component_system=None, name="unit_op", *args, **kwargs):
        if component_system is None:
            component_system = _two_component_system
        super().__init__(component_system, name, *args, **kwargs)

    def add_section(self, *args, **kwargs):
        pass

    def initialize_state(self) -> NoReturn:
        super().initialize_state()
        self.add_section()


class InletFixture(UnitOperationFixture, Inlet):
    def __init__(
        self,
        component_system=None,
        name="inlet",
        c_poly=None,
        viscosity=1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(component_system, name, *args, **kwargs)
        if c_poly is None:
            c_poly = np.arange(self.component_system.n_comp)
        self.c_poly = c_poly
        self.viscosity = viscosity

    def add_section(self, c_poly=None, start=0, end=1):
        if c_poly is None:
            c_poly = self.c_poly
        self.update_parameters(start, end, {"c_poly": c_poly})


class OutletFixture(UnitOperationFixture, Outlet):
    def __init__(self, component_system=None, name="outlet", *args, **kwargs):
        super().__init__(component_system, name, *args, **kwargs)


class CstrFixture(UnitOperationFixture, Cstr):
    def __init__(self, component_system=None, name="cstr", *args, **kwargs):
        super().__init__(component_system, name, *args, **kwargs)


class DeadEndFiltrationFixture(UnitOperationFixture, DeadEndFiltration):
    def __init__(
        self,
        component_system=None,
        name="dead_end_filtration",
        membrane_area=1,
        membrane_resistance=1,
        specific_cake_resistance=1,
        solution_viscosity=1,
        rejection_model=StepCutOff(cutoff_weight=0),
        viscosity_model=LogarithmicMixingViscosity(),
        *args,
        **kwargs,
    ):
        super().__init__(component_system, name, *args, **kwargs)
        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance
        self.specific_cake_resistance = specific_cake_resistance
        self.rejection_model = rejection_model
        self.solution_viscosity = solution_viscosity
        self.viscosity_model = viscosity_model


class CrossFlowFiltrationFixture(UnitOperationFixture, CrossFlowFiltration):
    def __init__(
        self,
        component_system=None,
        name="cross_flow_filtration",
        membrane_area=1,
        membrane_resistance=1e-9,
        *args,
        **kwargs,
    ):
        super().__init__(component_system, name, *args, **kwargs)
        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance


class _2DGRMFixture(UnitOperationFixture, _2DGRM):
    def __init__(self, component_system=None, name="2DGRM", *args, **kwargs):
        super().__init__(component_system, name, *args, **kwargs)


class SolveDummy(Structure):
    def __init__(self, test_dict=None):
        self._test_dict: Optional[Dict] = test_dict

    @property
    def values(self):
        return self._test_dict


class SystemFixture(FlowSystem):
    def __init__(self, unit_operations=None, *args, **kwargs):
        if unit_operations is None:
            unit_operations = [
                InletFixture(),
                DeadEndFiltrationFixture(),
                OutletFixture(),
            ]
        super().__init__(unit_operations, *args, **kwargs)
        self.initialize_state()


class SolverFixture(Solver):
    def __init__(self, system=None, sections=None, *args, **kwargs):
        if sections is None:
            sections = [
                {
                    "start": 0,
                    "end": 10,
                    "connections": [
                        ("inlet", "dead_end_filtration", 1e-3),
                        ("dead_end_filtration", "outlet", 1e-3),
                    ],
                    "section_states": {
                        "inlet": {"c_poly": [[1, 0, 0, 0], [2, 0, 0, 0]]}
                    },
                },
                {
                    "start": 10,
                    "end": 12,
                    "connections": [
                        ("inlet", "dead_end_filtration", 1e-3),
                        ("dead_end_filtration", "outlet", 0.5e-3),
                    ],
                    "section_states": {
                        "inlet": {"c_poly": [[0, 1, 0, 0], [1, 0, 0, 0]]}
                    },
                },
            ]
        if system is None:
            system = SystemFixture()
        super().__init__(system, sections, *args, **kwargs)
