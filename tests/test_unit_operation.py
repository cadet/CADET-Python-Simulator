from typing import NoReturn

import numpy as np
import pytest

from CADETPythonSimulator.componentsystem import CPSComponentSystem

from CADETPythonSimulator.unit_operation import (
    UnitOperationBase,
    Inlet, Outlet,
    Cstr,
    DeadEndFiltration, CrossFlowFiltration,
    _2DGRM
)

from CADETPythonSimulator.rejection import StepCutOff
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity


# %% Unit Operation Fixtures
class TwoComponentFixture(CPSComponentSystem):
    """Component Fixture class wiht two pre set components."""

    def __init__(self, *args, **kwargs):
        """Initialize the component with parameters."""
        super().__init__(*args, **kwargs)

        self.add_component(
            'A',
            molecular_weight=1e3,
            density=1e3,
            molecular_volume=1,
            viscosity=1
            )
        self.add_component(
            'B',
            molecular_weight=10e3,
            density=1e3,
            molecular_volume=1,
            viscosity=1
            )


class UnitOperationFixture(UnitOperationBase):
    """Unit Operation Fixture Class for testing purpose."""

    class_cps = TwoComponentFixture()
    def __init__(self, component_system, name, *args, **kwargs):
        """Initialize the unit operation."""
        if component_system is None:
            component_system = UnitOperationFixture.class_cps
        super().__init__(component_system, name, *args, **kwargs)

    def add_section(self, *args, **kwargs):
        """Add section depending on unit operation."""
        pass

    def initialize_state(self) -> NoReturn:
        """Initialize unit operation dependend for testing purpose."""
        super().initialize_state()
        self.add_section()

class InletFixture(UnitOperationFixture, Inlet):
    """Inlet fixture class for testing purpose, inherits from UnitOperationFixture."""

    def __init__(
            self,
            component_system=None,
            name='inlet',
            c_poly=None,
            viscosity=1e-3,
            *args,
            **kwargs
        ):
        """Initialize class for inlet fixture."""
        super().__init__(component_system, name, *args, **kwargs)

        if c_poly is None:
            c_poly = np.arange(self.component_system.n_comp)
        self.c_poly = c_poly

        self.viscosity = viscosity

    def add_section(self, c_poly=None, start=0, end=1):
        """For testing purpose, c_poly is set."""
        if c_poly is None:
            c_poly = self.c_poly
        self.update_parameters(start, end, {'c_poly': c_poly})


class OutletFixture(UnitOperationFixture, Outlet):
    """Oulet fixture class for testing purpose, inherits from UnitOperationFixture."""

    def __init__(self, component_system=None, name='outlet', *args, **kwargs):
        """Initialize the outlet fixture."""
        super().__init__(component_system, name, *args, **kwargs)


class CstrFixture(UnitOperationFixture, Cstr):
    """Cstr fixture class for testing purpose, inherits from UnitOperationFixture."""

    def __init__(self, component_system=None, name='cstr', *args, **kwargs):
        """Initialize the cstr fixture."""
        super().__init__(component_system, name, *args, **kwargs)


class DeadEndFiltrationFixture(UnitOperationFixture, DeadEndFiltration):
    """DEF fixture class for testing purpose, inherits from UnitOperationFixture."""

    def __init__(self,
                 component_system=None,
                 name='dead_end_filtration',
                 membrane_area=1,
                 membrane_resistance=1,
                 specific_cake_resistance=1,
                 solution_viscosity=1,
                 rejection_model=StepCutOff(cutoff_weight=0),
                 viscosity_model=LogarithmicMixingViscosity(),
                 *args,
                 **kwargs
        ):
        """Initialize DEF fixture with default parameter and default rejection."""
        super().__init__(component_system, name, *args, **kwargs)

        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance
        self.specific_cake_resistance = specific_cake_resistance
        self.rejection_model = rejection_model
        self.solution_viscosity = solution_viscosity
        self.viscosity_model = viscosity_model


class CrossFlowFiltrationFixture(UnitOperationFixture, CrossFlowFiltration):
    """CFF fixture class for testing purpose, inherits from UnitOperationFixture."""

    def __init__(self,
                 component_system=None,
                 name='cross_flow_filtration',
                 membrane_area=1,
                 membrane_resistance=1e-9,
                 *args,
                 **kwargs
        ):
        """Initialize CFF fixture with default parameter and defaultrejection."""
        super().__init__(component_system, name, *args, **kwargs)

        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance


class _2DGRMFixture(UnitOperationFixture, _2DGRM):
    def __init__(self,
                 component_system=None,
                 name='2DGRM',
                 *args,
                 **kwargs
                 ):
        super().__init__(component_system, name, *args, **kwargs)


# %% Unit Operation State Structure

@pytest.mark.parametrize(
    "unit_operation, expected",
    [
        (
            InletFixture(),
            {
                'n_inlet_ports': 0,
                'n_outlet_ports': 1,
                'n_dof': 2,
                'states': {
                    'outlet': [0., 1.],
                },
                'outlet_state': {
                    0: {
                        'c': [0., 1.],
                    },
                },
            },

        ),
        (
            OutletFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 0,
                'n_dof': 2,
                'states': {
                    'inlet': [0., 1.],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[:],
                        'value': [.1, .2],
                    },
                },
            },
        ),
        (
            CstrFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 1,
                'n_dof': 5,
                'states': {
                    'inlet': [0., 1.],
                    'bulk': [2., 3., 4.],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[:],
                        'value': [.1, .2, 2., 3., 4.],
                    },
                },
                'outlet_state': {
                    0: {
                        'c': [2., 3.],
                        'Volume': [4.],
                    },
                },
            },
        ),
        (
            DeadEndFiltrationFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 1,
                'n_dof': 15,
                'states': {
                    'cake': [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11],
                    'permeate_tank': [12., 13., 14.],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[0:12],
                        'value': [.1, .2, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11],
                    },
                },
                'outlet_state': {
                    0: {
                        'c': [12., 13.],
                        'tankvolume': [14.],
                    },
                },
            },
        ),
        (
            CrossFlowFiltrationFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 2,
                'n_dof': 80,
                'states': {
                    'retentate': [
                        [ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11],
                        [12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23],
                        [24, 25, 26, 27],
                        [28, 29, 30, 31],
                        [32, 33, 34, 35],
                        [36, 37, 38, 39],
                    ],
                    'permeate': [
                        [40, 41, 42, 43],
                        [44, 45, 46, 47],
                        [48, 49, 50, 51],
                        [52, 53, 54, 55],
                        [56, 57, 58, 59],
                        [60, 61, 62, 63],
                        [64, 65, 66, 67],
                        [68, 69, 70, 71],
                        [72, 73, 74, 75],
                        [76, 77, 78, 79],
                    ],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[0:10],
                        'value': [.1, .2, .3, 3., 4., 5., 6., 7., 8., 9.],
                    },
                },
                'outlet_state': {
                    0: {
                        'c': [36., 37.],
                        'viscosity': [38.],
                        'Volume': [39.],
                    },
                    1: {
                        'c': [76., 77.],
                        'viscosity': [78.],
                        'Volume': [79.],
                    },
                },
            },
        ),
        # (
        #     _2DGRMFixture(),
        #     {
        #         'n_inlet_ports': 3,
        #         'n_outlet_ports': 3,
        #         'n_dof': 900,
        #         'states': {
        #             'retentate': [
        #                 [ 0,  1,  2,  3],
        #                 [ 4,  5,  6,  7],
        #                 [ 8,  9, 10, 11],
        #                 [12, 13, 14, 15],
        #                 [16, 17, 18, 19],
        #                 [20, 21, 22, 23],
        #                 [24, 25, 26, 27],
        #                 [28, 29, 30, 31],
        #                 [32, 33, 34, 35],
        #                 [36, 37, 38, 39],
        #             ],
        #             'permeate': [
        #                 [40, 41, 42, 43],
        #                 [44, 45, 46, 47],
        #                 [48, 49, 50, 51],
        #                 [52, 53, 54, 55],
        #                 [56, 57, 58, 59],
        #                 [60, 61, 62, 63],
        #                 [64, 65, 66, 67],
        #                 [68, 69, 70, 71],
        #                 [72, 73, 74, 75],
        #                 [76, 77, 78, 79],
        #             ],
        #         },
        #     },
        # ),
    ]
)
class TestUnitStateStructure:
    """Test class for unit state structure."""

    def test_initialize(self, unit_operation: UnitOperationBase, expected: dict):
        """Check exception raising while not initialized."""
        with pytest.raises(Exception):
            unit_operation.states

    def test_state_structure(self, unit_operation: UnitOperationBase, expected: dict):
        """Initialize the unit operation and test structure."""
        unit_operation.initialize_state()
        assert unit_operation.n_inlet_ports == expected['n_inlet_ports']
        assert unit_operation.n_outlet_ports == expected['n_outlet_ports']
        assert unit_operation.n_dof == expected['n_dof']

    def test_states(self, unit_operation: UnitOperationBase, expected: dict):
        """Test state and directly state setting."""
        y_new = np.arange(unit_operation.n_dof, dtype=float)
        unit_operation.y = y_new

        for name, state in unit_operation.states.items():
            np.testing.assert_equal(state.s, expected['states'][name])

    def test_set_inlet_state(self, unit_operation: UnitOperationBase, expected: dict):
        """Test set_inlet_state_flat function."""
        s_in = {
            'c': [.1, .2],
        }
        if 'viscosity' in unit_operation.coupling_state_structure:
            s_in = {
                'c': [.1, .2],
                'viscosity': [.3],
        }
        if unit_operation.n_inlet_ports == 0:
            with pytest.raises(Exception):
                unit_operation.set_inlet_port_state(s_in, 0)
        else:
            for port in range(unit_operation.n_inlet_ports):
                unit_operation.set_inlet_state_flat(s_in, port)

                slice_information = expected['inlet_state'][port]
                state_slice = unit_operation.y[slice_information['slice']]
                np.testing.assert_array_equal(state_slice, slice_information['value'])

    def test_get_outlet_state(self, unit_operation: UnitOperationBase, expected: dict):
        """Test getter for outlet state."""
        if unit_operation.n_outlet_ports == 0:
            with pytest.raises(Exception):
                unit_operation.get_outlet_port_state(0)
        else:
            for port in range(unit_operation.n_outlet_ports):
                s_out = unit_operation.get_outlet_state_flat(port)
                np.testing.assert_equal(s_out, expected['outlet_state'][port])


@pytest.mark.parametrize(
    "unit_operation, case, residualfunc, expected",
    [
        # (
        #     InletFixture(),
        #     {
        #         'expected_residual': {
        #         },
        #     },
        # ),
        # (
        #     OutletFixture(),
        #     {
        #         'expected_residual': {
        #         },
        #     },
        # ),
        (
            CstrFixture(),
            {
                'states': {
                    'inlet': {
                        'c': np.array([7, 8]),
                    },
                    'bulk': {
                        'c': np.array([1, 2]),
                        'Volume': 1
                    }
                },
                'state_derivatives': {
                    'inlet': {
                        'c': [6, 7]
                    },
                    'bulk': {
                        'c': np.array([4, 5]),
                        'Volume': 2
                    }
                },
                'Q_in': [3],
                'Q_out': [4]
            },
            [
                (
                    "calculate_residual_concentration_cstr",
                    lambda c, c_dot, V, V_dot,  Q_in, Q_out, c_in:
                        c_dot * V + V_dot * c - Q_in * c_in + Q_out * c
                ),
                (
                    "calculate_residual_volume_cstr",
                    lambda V, V_dot, Q_in, Q_out: V_dot - Q_in + Q_out
                )
            ],
            {
                'inlet': {
                    'c': np.array([-7, -8])
                },
                'bulk': {
                    'c': np.array([-11, -7]),
                    'Volume': 3
                }
            }
        ),
        # (
        #      DeadEndFiltrationFixture(),
        #      {
        #         'states': {
        #             'cake': {
        #                 'c': np.array([0.5, 0.5]),
        #                 'pressure': 1,
        #                 'cakevolume': 1,
        #                 'permeate': 1,
        #             },
        #             'permeate': {
        #                 'c': np.array([0.5, 0.5]),
        #                 'Volume': 1,
        #             }
        #         },
        #         'state_derivatives': {
        #             'cake': {
        #                 'c': np.array([0.5, 0.5]),
        #                 'pressure': 1,
        #                 'cakevolume': 1,
        #                 'permeate': 1,
        #             },
        #             'permeate': {
        #                 'c': np.array([0.5, 0.5]),
        #                 'Volume': 1,
        #             }
        #         },
        #         'Q_in': [1],
        #         'Q_out': [1]
        #      },
        #      [
        #          ('CPSComponentSystem.molecular_weights', [1, 1]),
        #          ('CPSComponentSystem.molecular_volumes', [1, 1])
        #      ],
        #      {
        #         'cake': {
        #             'c': np.array([-0.5, -0.5]),
        #             'pressure': 2,
        #             'cakevolume': 0,
        #             'permeate': 0,
        #         },
        #         'permeate': {
        #             'c': np.array([1.5, 1.5]),
        #             'Volume': 2,
        #         }
        #      }
        # ),
        # (
        #     CrossFlowFiltrationFixture(),
        #     {
        #         'expected_residual': {
        #             },
        #     },
        # ),
        # (
        #     _2DGRMFixture(),
        #     {
        #         'expected_residual': {
        #             },
        #     },
        # ),
    ]
)
class TestUnitResidual():
    """Test class for resdiual related functions of unit operations."""

    def test_unit_residual(
            self,
            monkeypatch,
            unit_operation: UnitOperationBase,
            case: dict,
            residualfunc: dict,
            expected: dict
            ) -> NoReturn:
        """Test the residual of unit operations."""
        unit_operation.initialize_state()

        for funcname, func in residualfunc:
            monkeypatch.setattr('CADETPythonSimulator.unit_operation.'+funcname, func)

        for state, values in case['states'].items():
            for dof, new_value in values.items():
                unit_operation.states[state][dof] = new_value

        for state, values in case['state_derivatives'].items():
            for dof, new_value in values.items():
                unit_operation.state_derivatives[state][dof] = new_value

        unit_operation.Q_in = case['Q_in']
        unit_operation.Q_out = case['Q_out']

        unit_operation.compute_residual(3)

        for unit_module, module_dict in expected.items():
            for property, value in module_dict.items():
                np.testing.assert_equal(
                    value,
                    unit_operation.residuals[unit_module][property]
                )


# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_unit_operation.py"])
