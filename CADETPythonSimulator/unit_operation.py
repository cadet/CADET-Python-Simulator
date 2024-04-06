from abc import abstractmethod
from collections import defaultdict
from typing import Any, NoReturn

import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Typed, String, UnsignedInteger, UnsignedFloat, SizedUnsignedNdArray, NdPolynomial
)
from CADETProcess.dynamicEvents import Section

from CADETPythonSimulator.state import State, state_factory
from CADETPythonSimulator.rejection import RejectionBase
from CADETPythonSimulator.cake_compressibility import CakeCompressibilityBase


class UnitOperationBase(Structure):
    """
    Base class for unit operations.

    Attributes
    ----------
    name: str
        Name of the unit operation
    component_system : ComponentSystem
        Component system

    """

    name = String()
    component_system = Typed(ty=ComponentSystem)

    _states = []
    _parameters = []
    _section_dependent_parameters = []

    def __init__(self, component_system, name, *args, **kwargs):
        self.component_system = component_system
        self.name = name

        super().__init__(*args, **kwargs)

        self.parameter_sections = {}

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    def initialize_state(self) -> NoReturn:
        """Initialize the state."""
        _states = []
        for state in self._states:
            state_structure = getattr(self, state)
            state = state_factory(self, state, **state_structure)
            _states.append(state)
        self.__states = _states

    @property
    def states(self) -> list[State]:
        """The different state blocks of the unit operation."""
        return self.__states

    @property
    def states_dict(self) -> dict[str, State]:
        """The different state blocks of the unit operation, indexed by their name."""
        return {state.name: state for state in self.states}

    @property
    def n_dof(self) -> int:
        """int: Total number of degrees of freedom."""
        return sum([state.n_dof for state in self.states])

    @property
    def y_flat(self) -> np.ndarray:
        """np.ndarray: The state array flattened into one dimension."""
        return np.concatenate([state.y_flat for state in self.states])

    @y_flat.setter
    def y_flat(self, y_flat: np.ndarray) -> NoReturn:

        start_index = 0
        for state in self.states:
            end_index = start_index + state.n_dof
            state.y_flat = y_flat[start_index:end_index]
            start_index = end_index

    @property
    def y_split(self) -> dict[str, np.ndarray]:
        """dict: State arrays mapped to their state's names."""
        return {name: state for name, state in self.states_dict.items()}

    @y_split.setter
    def y_split(self, y_split: dict[str, np.ndarray]):
        for name, state in self.states_dict.items():
            state.y = y_split[name]

    @property
    def coupling_state_structure(self):
        """dict: State structure that must be accessible in inlet / outlet ports."""
        return {
            'c': self.n_comp,
            'viscosity': 1,
        }

    @property
    def n_dof_coupling(self) -> int:
        """int: Number of coupling DOFs."""
        return sum(self.coupling_state_structure.values())

    @property
    def inlet_ports(self) -> dict[str, int]:
        """dict: Number of inlet ports per state."""
        return {
            state.name: state.n_inlet_ports
            for state in self._states.values()
        }

    @property
    def n_inlet_ports(self) -> int:
        """int: Number of inlet ports."""
        return sum(state.n_inlet_ports for state in self.states)

    @property
    def outlet_ports(self) -> dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {
            state.name: state.n_outlet_ports
            for state in self._states.values()
        }

    @property
    def n_outlet_ports(self) -> int:
        """int: Number of inlet ports."""
        return sum(state.n_outlet_ports for state in self.states)

    @property
    def port_mapping(self) -> dict[int, str]:
        """dict: Mapping of port indices to corresponding state entries."""
        # TODO: Let this be handled by the SystemSolver?
        port_mapping = defaultdict(dict)

        counter = 0
        for mapped_state, n_ports in self.inlet_ports.items():
            for port in range(n_ports):
                port_mapping['inlet'][counter] = {}
                port_mapping['inlet'][counter]['mapped_state'] = mapped_state
                port_mapping['inlet'][counter]['port_index'] = port
                counter += 1

        counter = 0
        for mapped_state, n_ports in self.outlet_ports.items():
            for port in range(n_ports):
                port_mapping['outlet'][counter] = {}
                port_mapping['outlet'][counter]['mapped_state'] = mapped_state
                port_mapping['outlet'][counter]['port_index'] = port
                counter += 1

        return dict(port_mapping)

    def set_inlet_state(
            self,
            inlet_state: dict[str, np.ndarray],
            state: str,
            state_port_index: int
            ) -> NoReturn:
        """
        Set the state of the unit operation inlet for a given port.

        Parameters
        ----------
        inlet_state : Dict[str, np.ndarray]
            A dictionary mapping each state entry to its new values at the inlet port.
        state : str
            Name of the state for with to update the inlet state.
        state_port_index : int
            The state port index for which to update the state.

        Raises
        ------
        ValueError
            If state is not found.
        """
        if state not in self.states_dict:
            raise ValueError(f"Unknown state {state}.")

        self.states_dict[state].set_inlet_port_state(inlet_state, state_port_index)

    def set_inlet_state_flat(
            self,
            inlet_state: dict[str, np.ndarray],
            unit_port_index: int
            ) -> NoReturn:
        """
        Set the state of the unit operation inlet for a given port.

        Parameters
        ----------
        inlet_state : Dict[str, np.ndarray]
            A dictionary mapping each state entry to its new values at the inlet port.
        unit_port_index : int
            The index of the unit operation inlet port.
        """
        port_info = self.port_mapping['inlet'][unit_port_index]

        self.set_inlet_state(
            inlet_state, port_info['mapped_state'], port_info['port_index']
        )

    def get_outlet_state(
            self,
            state: str,
            state_port_index: int
            ) -> NoReturn:
        """
        Get the state of the unit operation outlet for a given port.

        Parameters
        ----------
        state : str
            Name of the state for with to retrieve the outlet state.
        state_port_index : int
            The state port index for which to retrieve the state.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping each state entry to the values at the outlet port.

        Raises
        ------
        ValueError
            If state is not found.
        """
        if state not in self.states_dict:
            raise ValueError(f"Unknown state {state}.")

        return self.states_dict[state].get_outlet_port_state(state_port_index)

    def get_outlet_state_flat(
            self,
            unit_port_index: int
            ) -> NoReturn:
        """
        Get the state of the unit operation outlet for a given port.

        Parameters
        ----------
        unit_port_index : int
            The index of the unit operation outlet port.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping each state entry to the values at the outlet port.
        """
        port_info = self.port_mapping['outlet'][unit_port_index]

        return self.get_inlet_state(port_info['mapped_state'], port_info['port_index'])

    @abstractmethod
    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> NoReturn:
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
        pass

    @property
    def section_dependent_parameters(self) -> list[str]:
        """list: Section depdendent parameters."""
        return self._section_dependent_parameters

    def update_section_dependent_parameters(
            self,
            start: float,
            end: float,
            parameters: dict[str, float | np.ndarray]
            ) -> NoReturn:
        """
        Update section dependent parameters.

        Parameters
        ----------
        start: float
            Start time of the section.
        end: float
            End time of the section.
        parameters : dict[str, float | np.ndarray]
            A dict with new parameters.

        Raises
        ------
        ValueError
            If not all section dependent parameters are provided.
        """
        if list(parameters.keys()) != self.section_dependent_parameters:
            raise ValueError(
                "All (and only) section dependent parameters must be provided."
            )

        for param, value in parameters.items():
            is_polynomial = param in self.polynomial_parameters
            setattr(self, param, value)
            coeffs = getattr(self, param)
            self.parameter_sections[param] = Section(start, end, coeffs, is_polynomial)

    def get_parameter_values_at_time(
            self,
            t: float,
            ) -> dict[str, np.typing.ArrayLike]:
        """
        Get parameter values at t.

        Parameters
        ----------
        t : float
            Current time.

        Returns
        -------
        dict
            Current values for each parameter.
        """
        if len(self.section_dependent_parameters) != len(self.parameter_sections):
            raise Exception("Section dependent parameters are not initialized.")

        current_parameters = {}
        for param in self.parameters:
            if param in self.section_dependent_parameters:
                value = self.parameter_sections[param](t)
            else:
                value = getattr(self, param)
            current_parameters[param] = value

        return current_parameters

    def __str__(self) -> str:
        """Return string represenation of the unit operation."""
        return self.name


class Inlet(UnitOperationBase):
    """
    System inlet.

    Attributes
    ----------
    c_poly : NdPolynomial
        Polynomial coefficients for component concentration.
    viscosity : float
        Viscosity of the solvent.
    """

    c_poly = NdPolynomial(size=('n_comp', 4), default=0)
    viscosity = UnsignedFloat()

    _parameters = ['c_poly', 'viscosity']
    _polynomial_parameters = ['c_poly']
    _section_dependent_parameters = ['c_poly']

    outlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1},
        'n_outlet_ports': 1,
    }
    _states = ['outlet']

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> NoReturn:
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
        # Inlet DOFs are simply copied to the residual.
        for i in range(self.n_dof_coupling):
            residual[i] = y[i]


class Outlet(UnitOperationBase):
    """System outlet."""

    inlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1},
        'n_inlet_ports': 1,
    }
    _states = ['inlet']

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> NoReturn:
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


class Cstr(UnitOperationBase):
    """
    Continuous stirred tank reactor.

    """

    inlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1},
        'n_inlet_ports': 1,
    }
    bulk = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'Volume': 1},
        'n_outlet_ports': 1,
    }
    _states = ['inlet', 'bulk']

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray,
            ) -> NoReturn:
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
        # Handle inlet DOFs, which are simply copied to the residual
        for i in range(self.n_inlet_ports):
            residual[i] = y[i]

        v = None
        v_dot = None
        offset = self.n_dof_coupling
        for i in range(self.n_dof):
            residual[offset + i] = y_dot[i] * v + v_dot * y[i]

        raise NotImplementedError()


class DeadEndFiltration(UnitOperationBase):
    """
    Dead end filtration model.

    Attributes
    ----------
    membrane_area : float
        Area of the membrane.
    membrane_resistance : float
        Membrane resistance.
    rejection_model : RejectionBase
        Model for size dependent rejection.
    cake_compressibility_model : CakeCompressibilityBase
        Model for cake compressibility.

    """
    retentate = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'Rc': 1, 'mc': 'n_comp'},
        'n_inlet_ports': 1,
    }
    permeate = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'Volume': 1},
        'n_outlet_ports': 1,
    }
    _states = ['retentate', 'permeate']

    rejection_model = Typed(ty=RejectionBase)
    cake_compressibility_model = Typed(ty=CakeCompressibilityBase)

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()

    _parameters = [
        'membrane_area',
        'membrane_resistance',
    ]

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
        raise self.cake_compressibility_model.specific_cake_resistance(delta_p)

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray
            ) -> NoReturn:
        #     0,  1,  2
        # y = Vp, Rc, mc
        # TODO: Needs to be extended to include c_in / c_out
        # y = [*c_i_in], viscosity_in, Vp, Rc, mc, [*c_i_out], viscosity_out

        c_in = y[0: self.n_comp]
        viscosity_in = y[self.n_comp]

        densities = self.component_system.densities

        residual[self.n_dof_coupling + 0] = ((self.membrane_area*self.delta_p(t)/viscosity_in)/(self.membrane_resistance+y[1])) - y_dot[0]
        residual[self.n_dof_coupling + 1] = (1/self.membrane_area) * (y_dot[2] * self.specific_cake_resistance(self.p(t))) - y_dot[1]

        residual[self.n_dof_coupling + 2] = ((self.c(t) * y_dot[0]) / (1-self.c(t)/self.density)) - y_dot[2]


class CrossFlowFiltration(UnitOperationBase):
    """
    Cross flow filtration model.

    Attributes
    ----------
    n_axial : int
        Number of axial discretization cells.
    membrane_area : float
        Area of the membrane.
    membrane_resistance : float
        Membrane resistance.
    rejection_model : RejectionBase
        Model for size dependent rejection.
    """

    n_axial = UnsignedInteger(default=10)

    retentate = {
        'dimensions': ('n_axial', ),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'Volume': 1},
        'n_inlet_ports': 1,
        'n_outlet_ports': 1,
    }
    permeate = {
        'dimensions': ('n_axial', ),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'Volume': 1},
        'n_outlet_ports': 1,
    }
    _states = ['retentate', 'permeate']

    rejection_model = Typed(ty=RejectionBase)

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()

    _parameters = [
        'membrane_area',
        'membrane_resistance',
    ]

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray
            ) -> NoReturn:
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


class _2DGRM(UnitOperationBase):
    """
    2D-General Rate Model.

    Attributes
    ----------
    n_axial : int
        Number of axial discretization cells.
    n_radial : int
        Number of radial discretization cells.
    n_particle : int
        Number of particle discretization cells.
    """

    n_axial = UnsignedInteger(default=10)
    n_radial = UnsignedInteger(default=3)
    n_particle = UnsignedInteger(default=5)

    bulk = {
        'dimensions': ('n_radial', 'n_axial'),
        'entries': {'c': 'n_comp', 'viscosity': 1},
        'n_inlet_ports': 'n_radial',
        'n_outlet_ports': 'n_radial',
    }
    particle = {
        'dimensions': ('n_radial', 'n_axial', 'n_particle'),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'q': 'n_comp'},
        'n_outlet_ports': 1,
    }
    flux = {
        'dimensions': ('n_radial', 'n_axial', ),
        'entries': {'c': 'n_comp'},
        'n_outlet_ports': 1,
    }
    _states = ['bulk', 'particle', 'flux']

    def compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            residual: np.ndarray
            ) -> NoReturn:
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
