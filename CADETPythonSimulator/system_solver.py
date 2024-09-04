from typing import NoReturn

from addict import Dict
import numpy as np
from scikits.odes.dae import dae

from CADETProcess.dataStructure import Structure
from CADETPythonSimulator.exception import NotInitializedError
from CADETPythonSimulator.unit_operation import UnitOperationBase


class SystemSolver(Structure):
    def __init__(self, unit_operations: list[UnitOperationBase], sections: list[dict]):
        self.initialize_solver()

        self._setup_unit_operations(unit_operations)
        self._setup_sections(sections)

    def _setup_unit_operations(self, unit_operations):
        self._unit_operations: list[UnitOperationBase] = unit_operations

        self.unit_slices: dict[UnitOperationBase, slice] = {}
        start_index = 0
        for unit in self.unit_operations:
            end_index = start_index + unit.n_dof
            self.unit_slices[unit] = slice(start_index, end_index)
            start_index = end_index

        # dict with [origin_index]{unit, port}
        origin_index_unit_operations = Dict()
        # Nested dict with [unit_operations][ports]: origin_index in connectivity matrix
        origin_unit_ports = Dict()
        origin_counter = 0
        for i_unit, unit in enumerate(self.unit_operations):
            for port in range(unit.n_outlet_ports):
                origin_unit_ports[i_unit][port] = origin_counter
                origin_index_unit_operations[origin_counter] = {
                    'unit': i_unit, 'port': port
                    }
                origin_counter += 1
        self.origin_unit_ports = origin_unit_ports
        self.origin_index_unit_operations = origin_index_unit_operations
        self.n_origin_ports = origin_counter

        # dict with [origin_index]{unit, port}
        destination_index_unit_operations = Dict()
        # Nested dict with [unit_operations][ports]: dest*_index in connectivity matrix
        destination_unit_ports = Dict()
        destination_counter = 0
        for i_unit, unit in enumerate(self.unit_operations):
            for port in range(unit.n_inlet_ports):
                destination_unit_ports[i_unit][port] = destination_counter
                destination_index_unit_operations[destination_counter] = {
                    'unit': i_unit, 'port': port
                    }
                destination_counter += 1
        self.destination_unit_ports = destination_unit_ports
        self.destination_index_unit_operations = destination_index_unit_operations
        self.n_destination_ports = destination_counter

    def _setup_sections(self, sections):
        # TODO: Check section continuity.

        self.sections = sections


    def initialize_solver(self, solver: str = 'ida') -> NoReturn:
        """
        Initialize solver.

        Parameters
        ----------
        solver : str, optional
            Solver to use for integration. The default is `ida`.

        """
        if solver not in ['ida']:
            raise ValueError(f"{solver} is not a supported solver.")

        self.solver = dae(solver, self.compute_residual)

    def initialize_solution_recorder(self) -> NoReturn:
        """
        Initialize the solution recorder for all unit_operations.

        Iterates over each unit in the system and initializes an empty numpy array for
        each state variable within the unit. The structure and size of the array for
        each state variable are determined by the unit's state structure.
        """
        self.unit_solutions: dict[UnitOperationBase, dict] = {}

        for unit in self.unit_operations:
            self.unit_solutions[unit]: dict[str, np.ndarray] = {}
            for state_name, state in unit.states_dict.items():
                self.unit_solutions[unit][state_name] = np.empty((0, *state.shape))
                self.unit_solutions[unit][f"{state_name}_dot"] = np.empty((0, *state.shape))

    def write_solution(self, y: np.ndarray, y_dot: np.ndarray) -> NoReturn:
        """
        Update the solution recorder for each unit with the current state.

        Iterates over each unit, to extract the relevant portions of `y` and `y_dot`.
        The current state of each unit is determined by splitting `y` according to each
        unit's requirements.

        Parameters
        ----------
        y : np.ndarray
            The current complete state of the system as a NumPy array.
        y_dot : np.ndarray
            The current complete derivative of the system's state as a NumPy array.

        """
        for unit, unit_slice in self.unit_slices.items():
            current_state = unit.y_split

            for state, value in current_state.items():
                previous_states = self.unit_solutions[unit][state]
                self.unit_solutions[unit][state] = np.vstack((
                    previous_states,
                    value.s.reshape((1, previous_states.shape[-1]))
                ))

            current_state_dot = unit.y_dot_split
            for state, value in current_state_dot.items():
                previous_states_dot = self.unit_solutions[unit][f"{state}_dot"]
                self.unit_solutions[unit][f"{state}_dot"] = np.vstack((
                    previous_states_dot,
                    value.s.reshape((1, previous_states_dot.shape[-1]))
                ))

    def solve(self) -> NoReturn:
        """Simulate the system."""
        self.initialize_solution_recorder()

        y_initial = self.y
        y_initial_dot = self.y_dot
        self.write_solution(y_initial, y_initial_dot)

        previous_end = self.sections[0].start
        for section in self.sections:
            if section.start <= section.end:
                raise ValueError("Section end must be larger than section start.")
            if section.start != previous_end:
                raise ValueError("Sections times must be without gaps.")

            self.update_section_states(
                section.start,
                section.end,
                section.section_states,
            )
            self.couple_unit_operations(
                section.connections,
                y_initial,
                y_initial_dot
            )

            section_solution_times = self.get_section_solution_times(section)
            y, y_dot = self.solve_section(
                section_solution_times, y_initial, y_initial_dot
            )

            self.write_solution(y, y_dot)

            y_initial = y
            y_initial_dot = y_dot

    def solve_section(
            self,
            section_solution_times: np.ndarray,
            y_initial: np.ndarray,
            y_initial_dot: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve a time section of the differential-algebraic equation system.

        This method uses the solver initialized by the system to compute the states and
        their derivatives over a specified section.

        Parameters
        ----------
        section_solution_times : np.ndarray
            The time points at which the solution is sought.
        y_initial : np.ndarray
            Initial values of the state variables at the start of the section.
        y_initial_dot : np.ndarray
            Initial derivatives of the state variables at the start of the section.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two NumPy arrays: the first array contains the computed
            values of the state variables (y) at each time point in
            `section_solution_times`, and the second array contains the derivatives of
            these state variables (y_dot) at each time point.

        """
        output = self.solver.solve(section_solution_times, y_initial, y_initial_dot)
        y = output.values.y
        y_dot = output.values.ydot

        return y, y_dot


    def get_section_solution_times(self, section: dict) -> np.ndarray:
        # TODO: How to get section_solution_times from section.start, section.end, if user_solution times are provided?
        raise NotImplementedError()

    def _update_section_states(
            self,
            start: float,
            end: float,
            section_states: dict[UnitOperationBase, dict]
            ) -> np.ndarray:
        """
        Update time dependent unit operation parameters.

        Parameters
        ----------
        start: float
            Start time of the section.
        end: float
            End time of the section.
        section_states : dict[UnitOperation, dict]
            Unit operation parameters for the next section.

        """
        for unit, parameters in section_states.items():
            # TODO: Check if unit is part of SystemSolver
            if isinstance(unit, str):
                unit = self.unit_operations_dict[unit]

            unit.update_section_dependent_parameters(start, end, parameters)

    # @property
    # def port_mapping(self) -> dict[int, str]:
    #     """dict: Mapping of port indices to corresponding state entries."""
    #     # TODO: Let this be handled by the SystemSolver?
    #     port_mapping = defaultdict(dict)

    #     counter = 0
    #     for mapped_state, n_ports in self.inlet_ports.items():
    #         for port in range(n_ports):
    #             port_mapping['inlet'][counter] = {}
    #             port_mapping['inlet'][counter]['mapped_state'] = mapped_state
    #             port_mapping['inlet'][counter]['port_index'] = port
    #             counter += 1

    #     counter = 0
    #     for mapped_state, n_ports in self.outlet_ports.items():
    #         for port in range(n_ports):
    #             port_mapping['outlet'][counter] = {}
    #             port_mapping['outlet'][counter]['mapped_state'] = mapped_state
    #             port_mapping['outlet'][counter]['port_index'] = port
    #             counter += 1

    #     return dict(port_mapping)

