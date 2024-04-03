import numpy as np
import pytest

from CADETProcess.processModel import ComponentSystem

from simulator import Inlet, Outlet, DeadEndFiltration

# %% Unit Operation Fixtures

component_system = ComponentSystem()
component_system.add_component(
    'A',
    molecular_weight=1e3,
    density=1e3,
)
component_system.add_component(
    'B',
    molecular_weight=10e3,
    density=1e3,
)

inlet = Inlet(component_system=component_system)
outlet = Outlet(component_system=component_system)

dead_end_filtration = DeadEndFiltration(component_system=component_system)

# %% Unit Operation State Structure

unit_state_structures = [
    {
        'unit': inlet,
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
        'spit_state': {
            'c_out': [0, 1],
            'viscosity_out': [2]
        }
    },
    {
        'unit': outlet,
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
        'spit_state': {
            'c_in': [0, 1],
            'viscosity_in': [2],
        }
    },
    {
        'unit': dead_end_filtration,
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
        'spit_state': {
            'c_in': [0, 1],
            'viscosity_in': [2],
            'Vp': [3],
            'Rc': [4],
            'mc': [5, 6],
            'c_out': [7, 8],
            'viscosity_out': [9]
        }
    },
    # TODO: Add some units with multiple ports.
]


@pytest.mark.parametrize("case", unit_state_structures)
def test_unit_state_structure(case):
    """Test the state of unit operations."""
    unit = case['unit']
    assert unit.inlet_state_structure == case['inlet_state_structure']
    assert unit.n_dof_inlet == case['n_dof_inlet']

    assert unit.internal_state_structure == case['internal_state_structure']
    assert unit.n_dof_internal == case['n_dof_internal']

    assert unit.outlet_state_structure == case['outlet_state_structure']
    assert unit.n_dof_outlet == case['n_dof_outlet']

    assert unit.state_structure == case['state_structure']
    assert unit.n_dof_total == case['n_dof_total']

    state = np.arange(unit.n_dof_total)
    current_state = unit.split_state(state)

    np.testing.assert_equal(current_state, case['spit_state'])


# %% TODO: Unit operation residual

# %% System Fixtures

from simulator import SystemSolver

units = [inlet, dead_end_filtration, outlet]

sections = [
    {
        'start': 0,
        'end': 10,
        'connections': [
            [0, 1, 1e-3],
            [1, 2, 1e-3],
        ],
        'section_states': {
            inlet: {
                'c': [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
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
            inlet: {
                'c': [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            },
        },
    },
]

system_solver = SystemSolver(units, sections)

# %% State Structure

system_state_structures = [
    {
        'system_solver': system_solver,
        'n_dof_system': 16,
        'unit_slices': {
            inlet: slice(0, 3, None),
            dead_end_filtration: slice(3, 13, None),
            outlet: slice(13, 16, None),
        },
        'unit_solution': {
            inlet: {
                'c_out': np.array([[0., 1.], [0., 2.]]),
                'c_out_dot': np.array([[ 0.,  2.], [0., 4.]]),
                'viscosity_out': np.array([[ 2.], [4.]]),
                'viscosity_out_dot': np.array([[4.], [8.]]),
            },
            dead_end_filtration: {
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
            outlet: {
                'c_in': np.array([[13., 14.], [26., 28.]]),
                'c_in_dot': np.array([[26., 28.], [52., 56.]]),
                'viscosity_in': np.array([[15.], [30.]]),
                'viscosity_in_dot': np.array([[30.], [60.]]),
            },
        },
    },
]


@pytest.mark.parametrize("case", system_state_structures)
def test_system_state_structure(case):
    """Test the system state."""
    system_solver = case['system_solver']

    assert system_solver.n_dof_system == case['n_dof_system']
    np.testing.assert_equal(system_solver.unit_slices, case['unit_slices'])

    system_solver.initialize_solution_recorder()

    solution = np.arange(system_solver.n_dof_system)
    system_solver.write_solution(solution, 2 * solution)
    system_solver.write_solution(2 * solution, 4 * solution)

    unit_solutions = system_solver.unit_solutions
    for unit, solutions in unit_solutions.items():
        for state, values in solutions.items():
            np.testing.assert_almost_equal(values, case['unit_solution'][unit][state])


# %% Section States

section_states = [
    {
          'section': sections[0],
          'expected_parameters': {
          },
    },
    {
          'section': sections[1],
          'expected_parameters': {
              inlet: {
                  'c': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
              },
          },
    },

]


@pytest.mark.parametrize("case", section_states)
def test_section_states(case):
    """Test the of unit operations."""
    section = case['section']
    system_solver._update_section_states(
        section['start'],
        section['end'],
        section_states=section['section_states'],
    )

    for unit, states in section['section_states'].items():
        for state, value in states.values():
            assert value
        assert unit.parameter_sections.keys() == states.keys()

        print(unit)
        # Assert all parameters are included
        # Assert all values are properly set in parameters
        # Assert sections were created accordingly


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

@pytest.mark.parametrize("case", connections)
def test_connections_matrix(case):
    matrix = system_solver._compute_connectivity_matrix(case['connections'])

    np.testing.assert_almost_equal(matrix, case['expected_matrix'])


@pytest.mark.parametrize("case", connections)
def test_coupling(case):
    y = np.arange(system_solver.n_dof_system)
    y_dot = 2 * y

    system_solver.couple_units(case['connections'], y, y_dot)
    np.testing.assert_almost_equal(y, case['expected_state'])

# %% TODO: System Residual



# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_simulator.py"])
