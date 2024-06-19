from typing import NoReturn

import numpy as np
import pytest

from CADETProcess.processModel import ComponentSystem

from CADETPythonSimulator.unit_operation import (
    UnitOperationBase,
    Inlet, Outlet,
    Cstr,
    DeadEndFiltration, CrossFlowFiltration,
    _2DGRM
)


# %% Unit Operation Fixtures
class TwoComponentFixture(ComponentSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_component('A', molecular_weight=1e3, density=1e3)
        self.add_component('B', molecular_weight=10e3, density=1e3)


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
                 membrane_resistance=1e-9,
                 *args,
                 **kwargs
                 ):
        super().__init__(component_system, name, *args, **kwargs)

        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance


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
                'y_split': {
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
                'y_split': {
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
                'y_split': {
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
                'y_split': {
                    'retentate': [0., 1., 2., 3., 4., 5.],
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
                'y_split': {
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
        #         'y_split': {
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

    def test_y_split(self, unit_operation, expected):
        y_new = np.arange(unit_operation.n_dof, dtype=float)
        unit_operation.y = y_new

        for name, state in unit_operation.y_split.items():
            np.testing.assert_equal(state.y, expected['y_split'][name])

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
    "unit_operation, case, expected",
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
                'y': [0, 0, 1, 2, 1],
                'y_dot': [0, 0, 0, 0, 0],
                'Q_in': 0,
                'Q_out': 0,
                't': 0,
            },
            {
                'residual': [0, 0, 0, 0, 0]
            },
        ),
        (
            CstrFixture(),
            {
                'y': [1, 2, 1, 2, 1],
                'y_dot': [0, 0, 0, 0, 0],
                'Q_in': 0,
                'Q_out': 0,
                't': 0,
            },
            {
                'residual': [1, 2, 0, 0, 0]
            },
        ),
        # (
        #     DeadEndFiltrationFixture(),
        #     {
        #         'expected_residual': {
        #             },
        #     },
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

    def test_unit_residual(
            self,
            unit_operation: UnitOperationBase,
            case: dict,
            expected: dict,
            ) -> NoReturn:
        """Test the residual of unit operations."""
        unit_operation.y = case['y']
        unit_operation.y_dot = case['y_dot']

        unit_operation.compute_residual(case['t'])

        np.testing.assert_almost_equal(unit_operation.residual, expected['residual'])


# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_unit_operation.py"])
