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


# %% Unit Operation Fixtures (pytest)
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

        self.initialize_state()

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

    def add_section(self, c=None, start=0, end=1):
        if c is None:
            c = self.c
        self.update_section_dependent_parameters(start, end, {'c': c})


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
                'state_structure': {
                    'outlet': {
                        'dimensions': (1,),
                        'structure': {'c': 2, 'viscosity': 1}
                    },
                },
                'n_dof': 3,
                'y_split': {
                    'outlet': [0, 1, 2],
                },
                'outlet_state': {
                    0: [3, 4, 5],
                },
            }
        ),
        (
            OutletFixture(),
            {
                'state_structure': {
                    'inlet': {
                        'dimensions': (1,),
                        'structure': {'c': 2, 'viscosity': 1}
                    },
                },
                'n_dof': 3,
                'y_split': {
                    'inlet': [0, 1, 2],
                },
            },
        ),
        (
            CstrFixture(),
            {
                'state_structure': {
                    'inlet': {
                        'dimensions': (1,),
                        'structure': {'c': 2, 'viscosity': 1},
                    },
                    'bulk': {
                        'dimensions': (1,),
                        'structure': {'c': 2, 'viscosity': 1, 'V': 1},
                    },
                },
                'n_dof': 7,
                'y_split': {
                    'inlet': [0, 1, 2],
                    'bulk': [3, 4, 5, 6],
                },
                'split_state': {
                    'inlet': {'c': [[0, 1]], 'viscosity': [[2]]},
                    'bulk': {'c': [[3, 4]], 'viscosity': [[5]], 'V': [[6]]},
                },
                'inlet_state': {
                    0: [0, 0, 0, 3, 4, 5, 6],
                },
                'outlet_state': {
                    0: [3, 4, 5],
                },
            },
        ),
        (
            DeadEndFiltrationFixture(),
            {
                'state_structure': {
                    'inlet': {
                        'dimensions': (1,),
                        'structure': {'c': 2, 'viscosity': 1}
                    },
                    'retentate': {
                        'dimensions': (1,),
                        'structure': {'Rc': 1, 'mc': 2, 'Vp': 1}
                    },
                    'permeate': {
                        'dimensions': (1,),
                        'structure': {'c': 2, 'viscosity': 1}
                    }
                },
                'n_dof': 10,
                'y_split': {
                    'retentate': [0, 1, 2, 3, 4, 5],
                    'permeate': [6, 7, 8, 9],
                },
                'split_state': {
                    'inlet': {'c': [[0, 1]], 'viscosity': [[2]]},
                    'retentate': {'Rc': [[3]], 'mc': [[4, 5]], 'Vp': [[6]]},
                    'permeate': {'c': [[7, 8]], 'viscosity': [[9]]},
                },
                'inlet_state': {
                    0: [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
                },
                'outlet_state': {
                    0: [7, 8, 9],
                },
            },
        ),
        (
            CrossFlowFiltrationFixture(),
            {
                'state_structure': {
                    'retentate': {
                        'dimensions': (10,),
                        'structure': {'c': 2, 'viscosity': 1, 'volume': 1},
                    },
                    'permeate': {
                        'dimensions': (10,),
                        'structure': {'c': 2, 'viscosity': 1, 'volume': 1},
                    },
                },
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
                'split_state': {
                    'retentate': {
                        'c': [
                            [ 0,  1],
                            [ 3,  4],
                            [ 6,  7],
                            [ 9, 10],
                            [12, 13],
                            [15, 16],
                            [18, 19],
                            [21, 22],
                            [24, 25],
                            [27, 28]
                        ],
                        'viscosity': [
                            [ 2],
                            [ 5],
                            [ 8],
                            [11],
                            [14],
                            [17],
                            [20],
                            [23],
                            [26],
                            [29]
                        ]
                    },
                    'permeate': {
                        'c': [
                            [30, 31],
                            [33, 34],
                            [36, 37],
                            [39, 40],
                            [42, 43],
                            [45, 46],
                            [48, 49],
                            [51, 52],
                            [54, 55],
                            [57, 58]
                        ],
                        'viscosity': [
                            [32],
                            [35],
                            [38],
                            [41],
                            [44],
                            [47],
                            [50],
                            [53],
                            [56],
                            [59]
                        ],
                    },
                },
                'inlet_state': {
                    0: [
                        0,  0,  0,  3,  4,  5,  6,  7,  8,  9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59
                    ],
                },
                'outlet_state': {
                    0: [30, 31, 32],
                    1: [57, 58, 59],
                },
            },
        ),
        # (
        #     _2DGRMFixture(),
        #     {
        #         'state_structure': {
        #             'bulk': {
        #                 'dimensions': (10, 3),
        #                 'structure': {'c': 2, 'viscosity': 1}
        #             },
        #             'particle': {
        #                 'dimensions': (10, 3, 5),
        #                 'structure': {'c': 2, 'viscosity': 1, 'q': 2}
        #             },
        #             'flux': {
        #                 'dimensions': (10, 3), 'structure': {'c': 2}
        #             },
        #         },
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

    def test_state_structure(self, unit_operation, expected):
        # assert unit_operation.state_structure == expected['state_structure']
        assert unit_operation.n_dof == expected['n_dof']

    def test_y_split(self, unit_operation, expected):
        y_new = np.arange(unit_operation.n_dof)
        unit_operation.y_flat = y_new

        for name, state in unit_operation.y_split.items():
            np.testing.assert_equal(state.y, expected['y_split'][name])

    def test_set_inlet_state(self, unit_operation, expected):
        if isinstance(unit_operation, Inlet):
            return
        y = np.arange(unit_operation.n_dof)
        s = np.ones(unit_operation.n_dof_coupling)
        for i_port in range(unit_operation.n_inlet_ports):
            unit_operation.set_inlet_state(y, i_port*s, i_port)
            np.testing.assert_equal(y, expected['inlet_state'][i_port])

    def test_get_outlet_state(self, unit_operation, expected):
        if isinstance(unit_operation, Outlet):
            return

        y = np.arange(unit_operation.n_dof)
        for i_port in range(unit_operation.n_inlet_ports):
            s = unit_operation.get_outlet_state(y, i_port)
            np.testing.assert_equal(s, expected['outlet_state'][i_port])

    def test_initialize_state(self, unit_operation, expected):
        unit_operation.initialize_state()

# %% TODO: Unit operation residual

@pytest.mark.parametrize(
    "unit_operation, expected",
    [
        (
            InletFixture(),
            {
                'expected_residual': {
                    },
            },
        ),
        (
            OutletFixture(),
            {
                'expected_residual': {
                    },
            },
        ),
        (
            CstrFixture(),
            {
                'expected_residual': {
                    },
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

    def test_unit_residual(self, unit_operation, expected):
        """Test the residual of unit operations."""
        unit_operation.add_section()

        y = np.arange(unit_operation.n_dof)
        y_dot = 2 * y
        residual = np.zeros((unit_operation.n_dof,))
        t = 0
        unit_operation.compute_residual(t, y, y_dot, residual)


# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_unit_operation.py"])
