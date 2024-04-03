from abc import abstractmethod
from typing import Tuple, Dict, List

from addict import Dict as ADict
import numpy as np
from scikits.odes.dae import dae

from CADETProcess.processModel import ComponentSystem
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Constant, Typed, UnsignedInteger, UnsignedFloat, SizedNdArray, NdPolynomial
)
from CADETProcess.dynamicEvents import Section


class UnitOperation(Structure):
    """
    Base class for unit operations.

    Attributes
    ----------
    initial_state : np.ndarray
        Initial state of unit operation
    component_system : ComponentSystem
        Component system
    inlet_ports : int
        Number of ports at the unit operation inlet.
    outlet_ports : int
        Number of ports at the unit operation outlet.
    """

    component_system = Typed(ty=ComponentSystem)
    initial_state = SizedNdArray(size='n_dof_total')
    initial_state_dot = SizedNdArray(size='n_dof_total')
    n_inlet_ports = UnsignedInteger(default=1)
    n_outlet_ports = UnsignedInteger(default=1)

    _parameters = []
    _section_dependent_parameters = []

    def __init__(self, component_system, *args, **kwargs):
        self.component_system = component_system
        super().__init__(*args, **kwargs)
        self.parameter_sections = {}

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    @property
    def inlet_state_structure(self):
        """
        Structure of the state at the unit operation inlet.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        if self.n_inlet_ports == 1:
            return {
                'c_in': self.n_comp,
                'viscosity_in': 1,
            }

        inlet_state_structure = {}
        for i in range(self.n_inlet_ports):
            inlet_state_structure[f'c_in_{i}'] = self.n_comp
            inlet_state_structure[f'viscosity_in_{i}'] = 1

        return inlet_state_structure

    @property
    def n_dof_inlet(self) -> int:
        """int: Number of degrees of freedom at the unit operation inlet."""
        return sum([n_states for n_states in self.inlet_state_structure.values()])

    @property
    def n_dof_inlet_port(self) -> int:
        """int: Number of degrees of freedom at a single unit operation inlet port."""
        return int(self.n_dof_inlet / self.n_inlet_ports)

    def set_inlet_state(
            self,
            y: np.ndarray,
            s: np.ndarray,
            port: int = 0,
            ) -> None:
        """
        Set the state of the unit operation inlet.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.
        s : np.ndarray
            New state of the unit operation inlet.
        port : int, optional
            Port of the unit operation for which to set the inlet state.
            Must be specified if unit operation has multiple inlet ports.
            The default is 0.

        Returns
        -------
        None.

        """
        if self.n_inlet_ports == 0:
            raise Exception(
                "Cannot set inlet state for unit operation without inlet ports."
            )
        if port > self.n_inlet_ports - 1:
            raise ValueError("Port exceeds number of inlet ports.")

        start_index = port * self.n_dof_inlet_port
        end_index = (port+1) * self.n_dof_inlet_port

        y[start_index:end_index] = s

    @property
    @abstractmethod
    def internal_state_structure(self) -> Dict[str, int]:
        """
        Internal structure of the unit operation state.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        return

    @property
    def n_dof_internal(self) -> int:
        """int: Number of degrees of freedom of the internal unit operation state."""
        return sum([n_states for n_states in self.internal_state_structure.values()])

    @property
    def outlet_state_structure(self):
        """
        Structure of the state at the unit operation outlet.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        if self.n_outlet_ports == 1:
            return {
                'c_out': self.n_comp,
                'viscosity_out': 1,
            }

        outlet_state_structure = {}
        for i in range(self.n_outlet_ports):
            outlet_state_structure[f'c_out_{i}'] = self.n_comp
            outlet_state_structure[f'viscosity_out_{i}'] = 1

        return outlet_state_structure

    @property
    def n_dof_outlet(self) -> int:
        """int: Total number of degrees of freedom at the unit operation outlet."""
        return sum([n_states for n_states in self.outlet_state_structure.values()])

    @property
    def n_dof_outlet_port(self) -> int:
        """int: Number of degrees of freedom at a single unit operation outlet port."""
        return int(self.n_dof_outlet / self.n_outlet_ports)

    def get_outlet_state(
            self,
            y: np.ndarray,
            port: int = 0,
            ) -> np.ndarray:
        """
        Return the state of the unit operation outlet.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.
        port : int, optional
            Port of the unit operation for which to return the outlet state.
            Only relevant if unit operation has multiple outlet ports.
            The default is 0.

        Returns
        -------
        np.ndarray
            State outlet state of the unit operation at given port.

        """
        if self.n_outlet_ports == 0:
            raise Exception(
                "Cannot retrieve outlet state for unit operation without outlet ports."
            )
        if port > self.n_outlet_ports - 1:
            raise ValueError("Port exceeds number of outlet ports.")

        start_index = -1 * (port+1) * self.n_dof_outlet_port
        end_index = -1 * (port) * self.n_dof_outlet_port

        return y[start_index:end_index if end_index != 0 else None]

    @property
    def state_structure(self) -> Dict[str, int]:
        """
        Structure of the unit state.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        return {
            **self.inlet_state_structure,
            **self.internal_state_structure,
            **self.outlet_state_structure,
        }

    def split_state(self, y) -> Dict[str, np.ndarray]:
        """
        Separate state into its components.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.

        Dict[str, np.ndarray]
            The state of the unit operation.
        """
        current_state = {}
        start_index = 0
        for state, size in self.state_structure.items():
            end_index = start_index + size
            current_state[state] = y[start_index:end_index]
            start_index = end_index

        return current_state

    @property
    def n_dof_total(self) -> int:
        """int: Number of degrees of freedom."""
        return sum([n_states for n_states in self.state_structure.values()])

    @abstractmethod
    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> None:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.
        y : np.ndarray
            Current state of the unit operation.
        y_dot : np.ndarray
            Current state derivative of the unit operation.
        residual : np.ndarray
            Residual of the unit operation.

        """
        return

    @property
    def section_dependent_parameters(self) -> List[str]:
        """list: Section depdendent parameters."""
        return self._section_dependent_parameters

    def update_section_dependent_parameters(
            self,
            start: float,
            end: float,
            parameters: Dict[str, float | np.ndarray]
            ) -> None:
        """
        Update section dependent parameters.

        Parameters
        ----------
        start: float
            Start time of the section.
        end: float
            End time of the section.
        parameters : Dict[str, float | np.ndarray]
            A dict with new parameters.

        Raises
        ------
        AttributeError
            If parameter cannot be found.
        ValueError
            If parameter cannot is not section dependent.
        """
        if len(self.parameter_sections) == 0:
            self.parameters_sections = {param: None for param in parameters}

        for param, value in parameters.items():
            if param not in self.parameters:
                raise AttributeError(f"Unknown parameter: {param}")
            if param not in self.section_dependent_parameters:
                raise ValueError(f"Parameter is not section dependent: {param}")
            setattr(self, param, value)

            is_polynomial = param in self.polynomial_parameters
            coeffs = getattr(self, param)
            self.parameter_sections[param] = Section(start, end, coeffs, is_polynomial)
            # TODO: Once defined via sections, a parameter must always be updated to
            # avoid inconsitent values. E.g. pop from dict


class Inlet(UnitOperation):
    """
    System inlet.

    Attributes
    ----------
    c : MultiTimeLine
        Piecewise cubic concentration profile.
    viscosity : float
        Viscosity of the solvent.
    """

    c = NdPolynomial(size=('n_comp', 4), default=0)
    viscosity = UnsignedFloat()

    n_inlet_ports = Constant(value=0)

    _parameters = ['c']
    _polynomial_parameters = ['c']
    _section_dependent_parameters = ['c']

    @property
    def internal_state_structure(self) -> Dict[str, int]:
        """
        Internal structure of the unit operation state.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        return {}

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> None:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.
        y : np.ndarray
            Current state of the unit operation.
        y_dot : np.ndarray
            Current state derivative of the unit operation.
        residual : np.ndarray
            Residual of the unit operation.

        """
        raise NotImplementedError()


class Outlet(UnitOperation):
    """System outlet."""

    n_outlet_ports = Constant(value=0)

    @property
    def internal_state_structure(self) -> Dict[str, int]:
        """
        Internal structure of the unit operation state.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        return {}

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> None:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.
        y : np.ndarray
            Current state of the unit operation.
        y_dot : np.ndarray
            Current state derivative of the unit operation.
        residual : np.ndarray
            Residual of the unit operation.

        """
        raise NotImplementedError()


class DeadEndFiltration(UnitOperation):
    mwco = UnsignedFloat()
    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()

    @property
    def internal_state_structure(self) -> Dict[str, int]:
        """
        Internal structure of the unit operation state.

        Returns a dictionary where each key is a state variable name and its value is
        the number of entries for that state.

        Returns
        -------
        Dict[str, int]
            The structure of the unit state.
        """
        return {
            'Vp': 1,
            'Rc': 1,
            'mc': self.n_comp,
        }

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray
            ) -> None:
        #     0,  1,  2
        # y = Vp, Rc, mc
        # TODO: Needs to be extended to include c_in / c_out
        # y = [*c_i_in], viscosity_in, Vp, Rc, mc, [*c_i_out], viscosity_out

        c_in = y[0: self.n_comp]
        viscosity_in = y[self.n_comp]

        density = self.component_system.density

        residual[self.n_dof_inlet + 0] = ((self.membrane_area*self.delta_p(t)/viscosity_in)/(self.membrane_resistance+y[1])) - y_dot[0]
        residual[self.n_dof_inlet + 1] = (1/self.membrane_area) * (y_dot[2] * self.specific_cake_resistance(self.p(t))) - y_dot[1]

        residual[self.n_dof_inlet + 2] = ((self.c(t) * y_dot[0]) / (1-self.c(t)/self.density)) - y_dot[2]

    def delta_p(self):
        raise NotImplementedError()

    def specific_cake_resistance(self, delta_p: float) -> float:
        """
        Compute specific resistance as a function of delta_p.

        Parameters
        ----------
        delta_p : float
            Pressure difference.

        Returns
        -------
        float
            Specific cake resistance.

        """
        raise NotImplementedError()


class SystemSolver():

    def __init__(self, units: List[UnitOperation], sections: List[dict]):
        self.initialize_solver()

        self._setup_units(units)
        self._setup_sections(sections)

    def _setup_units(self, units):
        self.units = units

        self.unit_slices = {}
        start_index = 0
        for unit in self.units:
            end_index = start_index + unit.n_dof_total
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
        self.sections = sections

    @property
    def n_dof_system(self) -> None:
        """int: Number of degrees of freedom of the system."""
        return sum([unit.n_dof_total for unit in self.units])

    def initialize_solver(self, solver: str = 'ida') -> None:
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

    def initialize_solution_recorder(self) -> None:
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

    def write_solution(self, y: np.ndarray, y_dot: np.ndarray) -> None:
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

    def solve(self) -> None:
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
            ) -> None:
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
            section_states: Dict[UnitOperation, dict]
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
            unit.update_section_dependent_parameters(start, end, parameters)
        self.previous_end = end

    def couple_units(
            self,
            connections: list,
            y_initial: np.ndarray,
            y_initial_dot: np.ndarray,
            ) -> None:
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

            s_new = np.zeros((destination_unit.n_dof_inlet_port,))
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
