import numpy as np
import pytest

from CADETProcess.processModel import ComponentSystem

from CADETPythonSimulator.simulator import Inlet, Outlet, Cstr, DeadEndFiltration

# %% Unit Operation Fixtures

# component_system = ComponentSystem()
# component_system.add_component(
#     'A',
#     molecular_weight=1e3,
#     density=1e3,
# )
# component_system.add_component(
#     'B',
#     molecular_weight=10e3,
#     density=1e3,
# )

# inlet = Inlet(component_system=component_system)
# outlet = Outlet(component_system=component_system)
# cstr = Cstr(component_system=component_system)
# dead_end_filtration = DeadEndFiltration(component_system=component_system)


# %% Unit Operation Fixtures (pytest)
class TwoComponentFixture(ComponentSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_component('A', molecular_weight=1e3, density=1e3)
        self.add_component('B', molecular_weight=10e3, density=1e3)


class UnitOperationFixture():
    def add_section(self, *args, **kwargs):
        pass


class InletFixture(UnitOperationFixture, Inlet):
    def __init__(
            self,
            component_system=None,
            name='inlet',
            c=None,
            viscosity=1e-3,
            *args,
            **kwargs
            ):
        if component_system is None:
            component_system = TwoComponentFixture()
        if c is None:
            c = np.arange(component_system.n_comp)

        super().__init__(
            component_system,
            name,
            c=c,
            viscosity=viscosity,
            *args, **kwargs
        )

    def add_section(self, c=None, start=0, end=1):
        if c is None:
            c = self.c
        self.update_section_dependent_parameters(start, end, {'c': c})


class OutletFixture(UnitOperationFixture, Outlet):
    def __init__(self, component_system=None, name='outlet', *args, **kwargs):
        if component_system is None:
            component_system = TwoComponentFixture()
        super().__init__(component_system, name, *args, **kwargs)


class CstrFixture(UnitOperationFixture, Cstr):
    def __init__(self, component_system=None, name='cstr', c=None, V=1, *args, **kwargs):
        if component_system is None:
            component_system = TwoComponentFixture()
        super().__init__(component_system, name, *args, **kwargs)

        if c is None:
            c = np.arange(self.n_comp)
        self.c = c

        self.V = V


class DeadEndFiltrationFixture(UnitOperationFixture, DeadEndFiltration):
    def __init__(self,
                 component_system=None,
                 name='dead_end_filtration',
                 c=None,
                 mwco=10e3,
                 membrane_area=1,
                 membrane_resistance=1e-9,
                 *args,
                 **kwargs
                 ):
        if component_system is None:
            component_system = TwoComponentFixture()
        super().__init__(component_system, name, *args, **kwargs)

        if c is None:
            c = np.arange(self.n_comp)
        self.c = c
        self.mwco = mwco
        self.membrane_area = membrane_area
        self.membrane_resistance = membrane_resistance


# %% Unit Operation State Structure

@pytest.mark.parametrize(
    "unit_operation, expected",
    [
        (
            InletFixture(),
            {
                'inlet_state_structure': {},
                'n_dof_inlet': 0,
                'internal_state_structure': {},
                'n_dof_internal': 0,
                'outlet_state_structure': {'c_out': 2, 'viscosity_out': 1},
                'n_dof_outlet': 3,
                'state_structure': {
                    'c_out': 2,
                    'viscosity_out': 1
                },
                'n_dof_total': 3,
                'split_state': {
                    'c_out': [0, 1],
                    'viscosity_out': [2]
                },
            }
        ),
        (
            OutletFixture(),
            {
                'inlet_state_structure': {'c_in': 2, 'viscosity_in': 1},
                'n_dof_inlet': 3,
                'internal_state_structure': {},
                'n_dof_internal': 0,
                'outlet_state_structure': {},
                'n_dof_outlet': 0,
                'state_structure': {
                    'c_in': 2,
                    'viscosity_in': 1,
                },
                'n_dof_total': 3,
                'split_state': {
                    'c_in': [0, 1],
                    'viscosity_in': [2],
                }
            },
        ),
        # (
        #     CstrFixture(),
        #     {
        #         'inlet_state_structure': {'c_in': 2, 'viscosity_in': 1},
        #         'n_dof_inlet': 3,
        #         'internal_state_structure': {},
        #         'n_dof_internal': 0,
        #         'outlet_state_structure': {},
        #         'n_dof_outlet': 0,
        #         'state_structure': {
        #             'c_in': 2,
        #             'viscosity_in': 1,
        #         },
        #         'n_dof_total': 3,
        #         'split_state': {
        #             'c_in': [0, 1],
        #             'viscosity_in': [2],
        #         },
        #     },
        # ),
        (
            DeadEndFiltrationFixture(),
            {
                'inlet_state_structure': {'c_in': 2, 'viscosity_in': 1},
                'n_dof_inlet': 3,
                'internal_state_structure': {'Vp': 1, 'Rc': 1, 'mc': 2},
                'n_dof_internal': 4,
                'outlet_state_structure': {'c_out': 2, 'viscosity_out': 1},
                'n_dof_outlet': 3,
                'state_structure': {
                    'c_in': 2,
                    'viscosity_in': 1,
                    'Vp': 1,
                    'Rc': 1,
                    'mc': 2,
                    'c_out': 2,
                    'viscosity_out': 1
                 },
                'n_dof_total': 10,
                'split_state': {
                    'c_in': [0, 1],
                    'viscosity_in': [2],
                    'Vp': [3],
                    'Rc': [4],
                    'mc': [5, 6],
                    'c_out': [7, 8],
                    'viscosity_out': [9]
                },
            },
        ),
    ]
)
class TestUnitStateStructure:

    def test_state_structure(self, unit_operation, expected):
        assert unit_operation.state_structure == expected['state_structure']
        assert unit_operation.n_dof_total == expected['n_dof_total']

    def test_inlet_state_structure(self, unit_operation, expected):
        assert unit_operation.inlet_state_structure == expected['inlet_state_structure']
        assert unit_operation.n_dof_inlet == expected['n_dof_inlet']

    def test_internal_state_structure(self, unit_operation, expected):
        assert unit_operation.internal_state_structure == expected['internal_state_structure']
        assert unit_operation.n_dof_internal == expected['n_dof_internal']

    def test_outlet_state_structure(self, unit_operation, expected):
        assert unit_operation.outlet_state_structure == expected['outlet_state_structure']
        assert unit_operation.n_dof_outlet == expected['n_dof_outlet']

    def test_split_state(self, unit_operation, expected):
        state = np.arange(unit_operation.n_dof_total)
        current_state = unit_operation.split_state(state)

        np.testing.assert_equal(current_state, expected['split_state'])


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
        (
            DeadEndFiltrationFixture(),
            {
                'expected_residual': {
                    },
            },
        ),
    ]
)
class TestUnitResidual():

    def test_unit_residual(self, unit_operation, expected):
        """Test the residual of unit operations."""
        unit_operation.add_section()

        y = np.arange(unit_operation.n_dof_total)
        y_dot = 2 * y
        residual = np.zeros((unit_operation.n_dof_total,))
        t = 0
        unit_operation.compute_residual(t, y, y_dot, residual)


# %% System Fixtures

from CADETPythonSimulator.simulator import SystemSolver


class SystemSolverFixture(SystemSolver):
    def __init__(self, unit_operations=None, sections=None, *args, **kwargs):
        if unit_operations is None:
            unit_operations = [
                InletFixture(),
                DeadEndFiltrationFixture(),
                OutletFixture()
            ]

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
                            'c': [
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
                            'c': [
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                            ],
                        },
                    },
                },
            ]

        super().__init__(unit_operations, sections, *args, **kwargs)


# %% State Structure

@pytest.mark.parametrize(
    "system_solver, expected",
    [
        (
            SystemSolverFixture(),
            {
                'n_dof_system': 16,
                'unit_slices': {
                    'inlet': slice(0, 3, None),
                    'dead_end_filtration': slice(3, 13, None),
                    'outlet': slice(13, 16, None),
                },
                'unit_solution': {
                    'inlet': {
                        'c_out': np.array([[0., 1.], [0., 2.]]),
                        'c_out_dot': np.array([[ 0.,  2.], [0., 4.]]),
                        'viscosity_out': np.array([[ 2.], [4.]]),
                        'viscosity_out_dot': np.array([[4.], [8.]]),
                    },
                    'dead_end_filtration': {
                        'c_in': np.array([[ 3.,  4.], [6., 8.]]),
                        'c_in_dot': np.array([[ 6.,  8.], [12., 16.]]),
                        'viscosity_in': np.array([[ 5.], [10.]]),
                        'viscosity_in_dot': np.array([[10.], [20.]]),
                        'Vp': np.array([[ 6.], [12.]]),
                        'Vp_dot': np.array([[12.], [24.]]),
                        'Rc': np.array([[7.], [14.]]),
                        'Rc_dot': np.array([[14.], [28.]]),
                        'mc': np.array([[8., 9.], [16., 18.]]),
                        'mc_dot': np.array([[16., 18.], [32., 36.]]),
                        'c_out': np.array([[10., 11.], [20., 22.]]),
                        'c_out_dot': np.array([[20., 22.], [40., 44.]]),
                        'viscosity_out': np.array([[12.], [24.]]),
                        'viscosity_out_dot': np.array([[24.], [48.]]),
                    },
                    'outlet': {
                        'c_in': np.array([[13., 14.], [26., 28.]]),
                        'c_in_dot': np.array([[26., 28.], [52., 56.]]),
                        'viscosity_in': np.array([[15.], [30.]]),
                        'viscosity_in_dot': np.array([[30.], [60.]]),
                    },
                },
                'section_states': {
                    0: {
                        'parameters_start': {
                            'inlet': {
                                'c': np.array([1., 2.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                        'parameters_end': {
                            'inlet': {
                                'c': np.array([1., 2.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                    },
                    1: {
                        'parameters_start': {
                            'inlet': {
                                'c': np.array([0., 1.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                        'parameters_end': {
                            'inlet': {
                                'c': np.array([2., 1.]),
                            },
                            'dead_end_filtration': {},
                            'outlet': {},
                        },
                    },
                },
            },
        ),
    ]
)
class TestSystemSolverStateStructure():

    def test_state_structure(self, system_solver, expected):
        """Test the system state."""
        assert system_solver.n_dof_system == expected['n_dof_system']
        # Transform keys to unit objects
        expected_slices = {
            system_solver.units_dict[unit]: value
            for unit, value in expected['unit_slices'].items()
        }
        np.testing.assert_equal(system_solver.unit_slices, expected_slices)

    def test_solution_recorder(self, system_solver, expected):
        """Test the solution recorder."""
        system_solver.initialize_solution_recorder()

        solution = np.arange(system_solver.n_dof_system)
        system_solver.write_solution(solution, 2 * solution)
        system_solver.write_solution(2 * solution, 4 * solution)

        unit_solutions = system_solver.unit_solutions
        for unit, solutions in unit_solutions.items():
            for state, values in solutions.items():
                np.testing.assert_almost_equal(
                    values, expected['unit_solution'][str(unit)][state]
                )

    def test_section_states(self, system_solver, expected):
        for i_sec, section in enumerate(system_solver.sections):
            system_solver._update_section_states(
                section['start'], section['end'], section['section_states']
            )

            for unit in system_solver.units:
                expected_values = expected['section_states'][i_sec]

                values_start = unit.get_current_parameter_values(section['start'])
                np.testing.assert_equal(
                    values_start, expected_values['parameters_start'][str(unit)]
                )

                values_end = unit.get_current_parameter_values(section['end'])
                np.testing.assert_equal(
                    values_end, expected_values['parameters_end'][str(unit)]
                )


# %% System Connectivity

connections = [
    {
        'connections': [[0, 1, 0, 0, 1e-3], [1, 2, 0, 0, 1e-3]],
        'expected_matrix': [
            [0.001, 0.   ],
            [0.   , 0.001]
        ],
        'expected_state': [0, 1, 2, 0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 10, 11, 12],
    },
    # TODO: Add tests with split/combined streams.
    # TODO: Add tests with recycling streams. => Iteration required?
    # TODO: Add tests with multiple ports.
]

# @pytest.mark.parametrize("case", connections)
# def test_connections_matrix(case):
#     matrix = system_solver._compute_connectivity_matrix(case['connections'])

#     np.testing.assert_almost_equal(matrix, case['expected_matrix'])


# @pytest.mark.parametrize("case", connections)
# def test_coupling(case):
#     y = np.arange(system_solver.n_dof_system)
#     y_dot = 2 * y

#     system_solver.couple_units(case['connections'], y, y_dot)
#     np.testing.assert_almost_equal(y, case['expected_state'])

# %% TODO: System Residual



# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_simulator.py"])
