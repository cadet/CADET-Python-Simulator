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


# %% Unit Operation Fixtures
class TwoComponentFixture(CPSComponentSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_component('A', molecular_weight=1e3, density=1e3, molecular_volume=1)
        self.add_component('B', molecular_weight=10e3, density=1e3, molecular_volume=1)


class UnitOperationFixture(UnitOperationBase):
    def __init__(self, component_system, name, *args, **kwargs):
        if component_system is None:
            component_system = TwoComponentFixture()
        super().__init__(component_system, name, *args, **kwargs)

        self.initialize()
        self.add_section()

    def add_section(self, *args, **kwargs):
        pass


class InletFixture(UnitOperationFixture, Inlet):
    def __init__(
            self,
            component_system=None,
            name='inlet',
            c_poly=None,
            viscosity=1e-3,
            *args,
            **kwargs
            ):
        super().__init__(component_system, name, *args, **kwargs)

        if c_poly is None:
            c_poly = np.arange(self.component_system.n_comp)
        self.c_poly = c_poly

        self.viscosity = viscosity

    def add_section(self, c_poly=None, start=0, end=1):
        if c_poly is None:
            c_poly = self.c_poly
        self.update_section_dependent_parameters(start, end, {'c_poly': c_poly})


class OutletFixture(UnitOperationFixture, Outlet):
    def __init__(self, component_system=None, name='outlet', *args, **kwargs):
        super().__init__(component_system, name, *args, **kwargs)


class CstrFixture(UnitOperationFixture, Cstr):
    def __init__(self, component_system=None, name='cstr', *args, **kwargs):
        super().__init__(component_system, name, *args, **kwargs)


class DeadEndFiltrationFixture(UnitOperationFixture, DeadEndFiltration):
    def __init__(self,
                 component_system=None,
                 name='dead_end_filtration',
                 membrane_area=1,
                 membrane_resistance=1,
                 specific_cake_resistance=1,
                 rejection=StepCutOff(cutoff_weight=0),
                 *args,
                 **kwargs
                 ):
        super().__init__(component_system, name, *args, **kwargs)

        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance
        self.specific_cake_resistance = specific_cake_resistance
        self.rejection = rejection


class CrossFlowFiltrationFixture(UnitOperationFixture, CrossFlowFiltration):
    def __init__(self,
                 component_system=None,
                 name='cross_flow_filtration',
                 membrane_area=1,
                 membrane_resistance=1e-9,
                 *args,
                 **kwargs
                 ):
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
                'n_dof': 3,
                'states': {
                    'outlet': [0., 1., 2.],
                },
                'outlet_state': {
                    0: {
                        'c': [0., 1.],
                        'viscosity': [2.]
                    },
                },
            },

        ),
        (
            OutletFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 0,
                'n_dof': 3,
                'states': {
                    'inlet': [0., 1., 2.],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[:],
                        'value': [.1, .2, .3],
                    },
                },
            },
        ),
        (
            CstrFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 1,
                'n_dof': 7,
                'states': {
                    'inlet': [0., 1., 2.],
                    'bulk': [3., 4., 5., 6.],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[:],
                        'value': [.1, .2, .3, 3., 4., 5., 6.],
                    },
                },
                'outlet_state': {
                    0: {
                        'c': [3., 4.],
                        'viscosity': [5.],
                        'Volume': [6.],
                    },
                },
            },
        ),
        (
            DeadEndFiltrationFixture(),
            {
                'n_inlet_ports': 1,
                'n_outlet_ports': 1,
                'n_dof': 10,
                'states': {
                    'cake': [0., 1., 2., 3., 4., 5.],
                    'permeate': [6., 7., 8., 9.],
                },
                'inlet_state': {
                    0: {
                        'slice': np.s_[:],
                        'value': [.1, .2, .3, 3., 4., 5., 6., 7., 8., 9.],
                    },
                },
                'outlet_state': {
                    0: {
                        'c': [6., 7.],
                        'viscosity': [8.],
                        'Volume': [9.],
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

    def test_state_structure(self, unit_operation: UnitOperationBase, expected: dict):
        assert unit_operation.n_inlet_ports == expected['n_inlet_ports']
        assert unit_operation.n_outlet_ports == expected['n_outlet_ports']
        assert unit_operation.n_dof == expected['n_dof']

    def test_states(self, unit_operation, expected):
        y_new = np.arange(unit_operation.n_dof, dtype=float)
        unit_operation.y = y_new

        for name, state in unit_operation.states.items():
            np.testing.assert_equal(state.s, expected['states'][name])

    def test_set_inlet_state(self, unit_operation: UnitOperationBase, expected: dict):
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
        if unit_operation.n_outlet_ports == 0:
            with pytest.raises(Exception):
                unit_operation.get_outlet_port_state(0)
        else:
            for port in range(unit_operation.n_outlet_ports):
                s_out = unit_operation.get_outlet_state_flat(port)
                np.testing.assert_equal(s_out, expected['outlet_state'][port])

    def test_initialize(self, unit_operation: UnitOperationBase, expected: dict):
        unit_operation.initialize()

# %% TODO: Unit operation residual


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
                        'viscosity': [3]
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
                    "calculate_residual_visc_cstr",
                    lambda *args: 0
                ),
                (
                    "calculate_residual_volume_cstr",
                    lambda V, V_dot, Q_in, Q_out: V_dot - Q_in + Q_out
                )
            ],
            {
                'inlet': {
                    'c': np.array([7, 8]),
                    'viscosity': 0
                },
                'bulk': {
                    'c': np.array([-11, -7]),
                    'Volume': 3
                }
            }
        ),
        (
             DeadEndFiltrationFixture(),
             {
                'states': {
                    'cake': {
                        'c': np.array([0.5, 0.5]),
                        'viscosity': 1,
                        'pressure': 1,
                        'cakevolume': 1,
                        'permeate': 1,
                    },
                    'permeate': {
                        'c': np.array([0.5, 0.5]),
                        'viscosity': 1,
                        'Volume': 1,
                    }
                },
                'state_derivatives': {
                    'cake': {
                        'c': np.array([0.5, 0.5]),
                        'viscosity': 1,
                        'pressure': 1,
                        'cakevolume': 1,
                        'permeate': 1,
                    },
                    'permeate': {
                        'c': np.array([0.5, 0.5]),
                        'viscosity': 1,
                        'Volume': 1,
                    }
                },
                'Q_in': [1],
                'Q_out': [1]
             },
             [
                 ('CPSComponentSystem.molecular_weights', [1, 1]),
                 ('CPSComponentSystem.molecular_volumes', [1, 1])
             ],
             {
                'cake': {
                    'c': np.array([0.5, 0.5]),
                    'viscosity': 0,
                    'pressure': 1,
                    'cakevolume': 0,
                    'permeate': 1,
                },
                'permeate': {
                    'c': np.array([1.5, 1.5]),
                    'viscosity': 0,
                    'Volume': 1,
                }
             }
        ),
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

    def test_unit_residual(
            self,
            monkeypatch,
            unit_operation: UnitOperationBase,
            case: dict,
            residualfunc: dict,
            expected: dict
            ) -> NoReturn:
        """Test the residual of unit operations."""

        for funcname, func in residualfunc:
            monkeypatch.setattr('CADETPythonSimulator.unit_operation.'+funcname, func)

        for key, value in case['states'].items():
            unit_operation.states[key] = value

        for key, value in case['state_derivatives'].items():
            unit_operation.state_derivatives[key] = value

        unit_operation._Q_in = case['Q_in']
        unit_operation._Q_out = case['Q_out']

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
