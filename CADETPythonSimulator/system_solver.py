from typing import Tuple, Dict, List, NoReturn

from addict import Dict as ADict
import numpy as np
from scikits.odes.dae import dae

from CADETProcess.dataStructure import Structure
# from CADETProcess.dataStructure import ()
from CADETPythonSimulator.unit_operation import UnitOperationBase


class SystemSolver(Structure):
    def __init__(self, units: List[UnitOperationBase], sections: List[dict]):
        self.initialize_solver()

        self._setup_units(units)
        self._setup_sections(sections)

    def _setup_units(self, units):
        self.units = units

        self.units_dict = {}
        for unit in self.units:
            self.units_dict[str(unit)] = unit

        self.unit_slices = {}
        start_index = 0
        for unit in self.units:
            end_index = start_index + unit.n_dof
            self.unit_slices[unit] = slice(start_index, end_index)
            start_index = end_index

        # Dict with [origin_index]{unit, port}
        origin_index_units = ADict()
        # Nested dict with [units][ports]: origin_index in connectivity matrix
        origin_unit_ports = ADict()
        origin_counter = 0
        for i_unit, unit in enumerate(self.units):
            for port in range(unit.n_outlet_ports):
                origin_unit_ports[i_unit][port] = origin_counter
                origin_index_units[origin_counter] = {'unit': i_unit, 'port': port}
                origin_counter += 1
        self.origin_unit_ports = origin_unit_ports
        self.origin_index_units = origin_index_units
        self.n_origin_ports = origin_counter

        # Dict with [origin_index]{unit, port}
        destination_index_units = ADict()
        # Nested dict with [units][ports]: destination_index in connectivity matrix
        destination_unit_ports = ADict()
        destination_counter = 0
        for i_unit, unit in enumerate(self.units):
            for port in range(unit.n_inlet_ports):
                destination_unit_ports[i_unit][port] = destination_counter
                destination_index_units[destination_counter] = {'unit': i_unit, 'port': port}
                destination_counter += 1
        self.destination_unit_ports = destination_unit_ports
        self.destination_index_units = destination_index_units
        self.n_destination_ports = destination_counter

    def _setup_sections(self, sections):
        # TODO: Check section continuity.

        self.sections = sections

    @property
    def n_dof_system(self) -> NoReturn:
        """int: Number of degrees of freedom of the system."""
        return sum([unit.n_dof for unit in self.units])

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
        Initialize the solution recorder for all units.

        Iterates over each unit in the system and initializes an empty numpy array for
        each state variable within the unit. The structure and size of the array for
        each state variable are determined by the unit's state structure.
        """
        self.unit_solutions = ADict()

        for unit in self.units:
            for state, size in unit.state_structure.items():
                self.unit_solutions[unit][state] = np.empty((0, size))
                self.unit_solutions[unit][f"{state}_dot"] = np.empty((0, size))

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
            current_state = unit.split_state(y[unit_slice])

            for state, value in current_state.items():
                previous_states = self.unit_solutions[unit][state]
                self.unit_solutions[unit][state] = np.vstack((
                    previous_states,
                    value.reshape((1, previous_states.shape[-1]))
                ))

            current_state_dot = unit.split_state(y_dot[unit_slice])
            for state, value in current_state_dot.items():
                previous_states_dot = self.unit_solutions[unit][f"{state}_dot"]
                self.unit_solutions[unit][f"{state}_dot"] = np.vstack((
                    previous_states_dot,
                    value.reshape((1, previous_states_dot.shape[-1]))
                ))

    def solve(self) -> NoReturn:
        """Simulate the system."""
        self.initialize_solution_recorder()

        y_initial, y_initial_dot = self.get_initial_conditions()
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
            self.couple_units(
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

    def get_initial_conditions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather initial conditions for all unit operations.

        Iterates over the units in the system, collecting their initial states
        and initial derivatives of states, and returns them as NumPy arrays.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two NumPy arrays: the first for initial states (y0)
            and the second for the derivatives of these states (y0dot).
        """
        y0 = [unit.initial_state for unit in self.units]
        y0dot = [unit.initial_state_dot for unit in self.units]

        return np.array(y0), np.array(y0dot)

    def solve_section(
            self,
            section_solution_times: np.ndarray,
            y_initial: np.ndarray,
            y_initial_dot: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
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
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two NumPy arrays: the first array contains the computed
            values of the state variables (y) at each time point in
            `section_solution_times`, and the second array contains the derivatives of
            these state variables (y_dot) at each time point.
        """
        output = self.solver.solve(section_solution_times, y_initial, y_initial_dot)
        y = output.values.y
        y_dot = output.values.ydot

        return y, y_dot

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray
            ) -> NoReturn:
        """
        Compute the residual for the differential-algebraic equations system.

        This method iterates over each unit in the system and computes the residuals
        based on the current state (y), the rate of change of the state (y_dot), and the
        current time (t). The computed residuals are directly updated in the passed
        `residual` array.

        Parameters
        ----------
        t : float
            Current time point.
        y : np.ndarray
            Current values of the state variables.
        y_dot : np.ndarray
            Current rates of change of the state variables.
        residual : np.ndarray
            Array to be filled with the computed residuals.

        """
        for unit, unit_slice in self.unit_slices.items():
            residual[unit_slice] = unit.compute_residual(
                t,
                y[unit_slice],
                y_dot[unit_slice],
                residual[unit_slice]
            )

    def get_section_solution_times(self, section: Dict) -> np.ndarray:
        # TODO: How to get section_solution_times from section.start, section.end, if user_solution times are provided?
        raise NotImplementedError()

    def _update_section_states(
            self,
            start: float,
            end: float,
            section_states: Dict[UnitOperationBase, dict]
            ) -> np.ndarray:
        """
        Update time dependent unit operation parameters.

        Parameters
        ----------
        start: float
            Start time of the section.
        end: float
            End time of the section.
        section_states : Dict[UnitOperation, dict]
            Unit operation parameters for the next section.

        """
        for unit, parameters in section_states.items():
            # TODO: Check if unit is part of SystemSolver
            if isinstance(unit, str):
                unit = self.units_dict[unit]

            unit.update_section_dependent_parameters(start, end, parameters)

    def couple_units(
            self,
            connections: list,
            y_initial: np.ndarray,
            y_initial_dot: np.ndarray,
            ) -> NoReturn:
        """
        Couple unit operations for a given section.

        Parameters
        ----------
        connections : list
            Flow sheet connectivity of the section.
        y_initial : np.ndarray
            Initial state of the sections.
        y_initial_dot : np.ndarray
            Initial state derivative of the sections.
        """
        connectivity_matrix = self._compute_connectivity_matrix(connections)

        for destination_port_index, Q_destinations in enumerate(connectivity_matrix):
            Q_destination_total = sum(Q_destinations)
            if Q_destination_total == 0:
                continue

            destination_info = self.destination_index_units[destination_port_index]
            destination_unit = self.units[destination_info['unit']]
            destination_port = destination_info['port']

            s_new = np.zeros((destination_unit.n_dof_coupling,))
            for origin_port_index, Q_destination in enumerate(Q_destinations):
                if Q_destination == 0:
                    continue

                origin_info = self.origin_index_units[origin_port_index]
                origin_unit = self.units[origin_info['unit']]
                origin_port = origin_info['port']

                y_origin_unit = y_initial[self.unit_slices[origin_unit]]
                s_unit = origin_unit.get_outlet_state(y_origin_unit, origin_port)

                s_new += s_unit * Q_destination  # Accumulate weighted states

            s_new /= Q_destination_total  # Normalize by total flow rate

            y_destination_unit = y_initial[self.unit_slices[destination_unit]]
            destination_unit.set_inlet_state(y_destination_unit, s_new, destination_port)

    def _compute_connectivity_matrix(self, connections: list) -> np.ndarray:
        """
        Compute the connectivity matrix from the connections interface.

        # Connectivity with ports
        ```
        'connections': [
            [0, 1, 0, 0, 1e-3], # unit 0 to unit 1, port 0 to port 0
            [1, 2, 0, 0, 0.5e-3], # unit 1 to unit 2, port 0 to port 0 (e.g. retentate)
            [1, 3, 1, 0, 0.5e-3], # unit 1 to unit 3, port 1 to port 0 (e.g. permeate)
        ]
        ```
        would translate to:
        ```
        connections = [
        # from  0  1/0 1/1 2  3   (to)
               [0, 0,  0,  0, 0],  # 0
               [1, 0,  0,  0, 0],  # 1
               [0, 1,  0,  0, 0],  # 2
               [0, 0,  1,  0, 0],  # 3
        ]
        ```

        Parameters
        ----------
        connections : list
            Flow sheet connectivity of the section.

        Returns
        -------
        np.ndarray
            Connectivity matrix.
        """
        connections = np.asarray(connections)

        connections_matrix = np.zeros((self.n_destination_ports, self.n_origin_ports))
        for connection in connections:
            origin_unit = connection[0]
            origin_port = connection[2]
            origin_index = self.origin_unit_ports[origin_unit][origin_port]

            destination_unit = connection[1]
            destination_port = connection[3]
            destination_index = self.destination_unit_ports[destination_unit][destination_port]

            flow_rate = connection[4]

            connections_matrix[destination_index, origin_index] = flow_rate

        return connections_matrix
