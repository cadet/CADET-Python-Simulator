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

from CADETPythonSimulator.exception import NotInitializedError, CADETPythonSimError
from CADETPythonSimulator.state import State, state_factory
from CADETPythonSimulator.residual import (
    calculate_residual_volume_cstr, 
    calculate_residual_concentration_cstr, 
    calculate_residual_visc_cstr,
    calculate_residual_press_easy_def,
    calculate_residual_cake_vol_def,
    calculate_residual_perm_easy_def,
    calculate_residual_visc_def
    )
from CADETPythonSimulator.rejection import RejectionBase
from CADETPythonSimulator.cake_compressibility import CakeCompressibilityBase
from CADETPythonSimulator.viscosity import LogarithmicMixingViscosity, ViscosityBase


class UnitOperationBase(Structure):
    """
    Base class for unit operations.

    Attributes
    ----------
    component_system : ComponentSystem
        Component system
    name: str
        Name of the unit operation

    """

    name = String()
    component_system = Typed(ty=ComponentSystem)

    _state_structures = []
    _parameters = []
    _section_dependent_parameters = []

    def __init__(self, component_system, name, *args, **kwargs):
        self.component_system = component_system
        self.name = name

        super().__init__(*args, **kwargs)

        self.parameter_sections = {}

        self._states = None
        self._state_derivatives = None
        self._residuals = None

        self._Q_in = None
        self._Q_out = None

    @property
    def n_dof(self) -> int:
        """int: Total number of degrees of freedom."""
        return sum([state.n_dof for state in self.states.values()])

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    def initialize(self) -> NoReturn:
        """Initialize the unit operation state and residual."""
        self._states = {}
        self._state_derivatives = {}
        self._residuals = {}
        for state_block in self._state_structures:
            state_structure = getattr(self, state_block)

            state = state_factory(self, state_block, **state_structure)
            self._states[state.name] = state

            state_derivative = state_factory(self, state_block, **state_structure)
            self._state_derivatives[state_derivative.name] = state_derivative

            residual = state_factory(self, state_block, **state_structure)
            self._residuals[residual.name] = residual

        self._Q_in = np.zeros(self.n_inlet_ports)
        self._Q_out = np.zeros(self.n_outlet_ports)

    @property
    def states(self) -> dict[str, State]:
        """dict: State array blocks of the unit operation, indexed by name."""
        if self._states is None:
            raise NotInitializedError("Unit operation state is not yet initialized.")

        return self._states

    @property
    def y(self) -> np.ndarray:
        """np.ndarray: State array flattened into one dimension."""
        return np.concatenate([state.s_flat for state in self.states.values()])

    @y.setter
    def y(self, y: np.ndarray) -> NoReturn:
        start_index = 0
        for state in self.states.values():
            end_index = start_index + state.n_dof
            state.s_flat = y[start_index:end_index]
            start_index = end_index

    @property
    def state_derivatives(self) -> dict[str, State]:
        """dict: State derivative array blocks of the unit operation, indexed by name."""
        if self._state_derivatives is None:
            raise NotInitializedError("Unit operation state is not yet initialized.")

        return self._state_derivatives

    @property
    def y_dot(self) -> np.ndarray:
        """np.ndarray: State derivative array flattened into one dimension."""
        return np.concatenate([
                state_derivative.s_flat
                for state_derivative in self.state_derivatives.values()
        ])

    @y_dot.setter
    def y_dot(self, y_dot: np.ndarray) -> NoReturn:
        start_index = 0
        for state_derivative in self.state_derivatives.values():
            end_index = start_index + state_derivative.n_dof
            state_derivative.s_flat = y_dot[start_index:end_index]
            start_index = end_index

    @property
    def residuals(self) -> list[State]:
        """list: Residual array blocks of the unit operation."""
        if self._residuals is None:
            raise NotInitializedError("Unit operation residual is not yet initialized.")

        return self._residuals

    @property
    def r(self) -> np.ndarray:
        """np.ndarray: Residual array flattened into one dimension."""
        return np.concatenate([
            residual.s_flat for residual in self.residuals.values()
        ])

    @r.setter
    def r(self, r: np.ndarray) -> NoReturn:
        start_index = 0
        for residual in self.residuals.values():
            end_index = start_index + residual.n_dof
            residual.s_flat = r[start_index:end_index]
            start_index = end_index

    @property
    def Q_in(self) -> np.ndarray:
        """np.ndarray: Ingoing flow rates."""
        if self._Q_in is None:
            raise NotInitializedError(
                "Unit operation flow rates are not yet initialized."
            )

        return self._Q_in

    @property
    def Q_out(self) -> np.ndarray:
        """np.ndarray: Ingoing flow rates."""
        if self._Q_out is None:
            raise NotInitializedError(
                "Unit operation flow rates are not yet initialized."
            )

        return self._Q_out

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
            state.name: state.n_inlet_ports for state in self.states.values()
        }

    @property
    def n_inlet_ports(self) -> int:
        """int: Number of inlet ports."""
        return sum(state.n_inlet_ports for state in self.states.values())

    @property
    def outlet_ports(self) -> dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {
            state.name: state.n_outlet_ports for state in self.states.values()
        }

    @property
    def n_outlet_ports(self) -> int:
        """int: Number of inlet ports."""
        return sum(state.n_outlet_ports for state in self.states.values())

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
        try:
            state = self.states[state]
        except KeyError:
            raise ValueError(f"Unknown state {state}.")

        state.set_inlet_port_state(inlet_state, state_port_index)

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
        try:
            state = self.states[state]
        except KeyError:
            raise ValueError(f"Unknown state {state}.")

        return state.get_outlet_port_state(state_port_index)

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

        return self.get_outlet_state(port_info['mapped_state'], port_info['port_index'])

    @abstractmethod
    def compute_residual(
            self,
            t: float,
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
    _state_structures = ['outlet']

    def compute_residual(
            self,
            t: float,
            residual: np.ndarray,
            ) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.
        residual : np.ndarray
            Residual of the unit operation.

        """
        # Inlet DOFs are simply copied to the residual.

        self.residuals['outlet']['c_poly'] = self.states['outlet']['c_poly']
        self.residuals['outlet']['viscosity'] = self.states['outlet']['viscosity']
        self.residuals['outlet']['c_poly'] = self.states['outlet']['c_poly']
        self.residuals['outlet']['viscosity'] = self.states['outlet']['viscosity']

class Outlet(UnitOperationBase):
    """System outlet."""

    inlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1},
        'n_inlet_ports': 1,
    }
    _state_structures = ['inlet']

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
    _state_structures = ['inlet', 'bulk']

    def compute_residual(
            self,
            t: float,
            ) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.
        """
        c_in = self.states['inlet']['c']
        c_in_dot = self.state_derivatives['inlet']['c']

        viscosity_in = self.states['inlet']['viscosity']

        c = self.states['bulk']['c']
        c_dot = self.state_derivatives['bulk']['c']

        V = self.states['bulk']['Volume']
        V_dot = self.state_derivatives['bulk']['Volume']

        # Handle inlet DOFs, which are simply copied to the residual
        self.residuals['inlet']['c'] = c_in

        # Handle bulk/outlet DOFs
        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]

        # for i in range(self.n_comp):
        #     self.residuals['bulk']['c'][i] = c_dot[i] * V + V_dot * c[i] - Q_in * c_in[i] + Q_out * c[i]
        # Alternative: Can we vectorize this?
        self.residuals['bulk']['c'] = calculate_residual_concentration_cstr(c, c_dot, V, V_dot,  Q_in, Q_out, c_in)

        self.residuals['bulk']['Volume'] = calculate_residual_volume_cstr(V, V_dot, Q_in, Q_out)

        self.residuals['inlet']['viscosity'] = calculate_residual_visc_cstr()

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
    cake = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'pressure': 1, 'cakevolume': 1, 'permeate': 1},
        'n_inlet_ports': 1,
    }
    bulk = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'viscosity': 1, 'Volume': 1},
        'n_outlet_ports': 1,
    }
    _state_structures = ['cake', 'bulk']

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()
    specific_cake_resistance = UnsignedFloat()
    molar_volume = SizedUnsignedNdArray(size = 'n_comp')
    efficency = SizedUnsignedNdArray(size = 'n_comp')

    _parameters = [
        'membrane_area',
        'membrane_resistance',
        'specific_cake_resistance',
        'molar_volume',
        'efficency'
    ]

    def compute_residual(
            self,
            t: float,
            ) -> NoReturn:        

        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]

        c_in = self.states['cake']['c']
        c_in_dot = self.state_derivatives['cake']['c']

        V_C = self.states['cake']['cakevolume']
        V_dot_C = self.state_derivatives['cake']['cakevolume']

        V_p = self.states['cake']['permeate']
        Q_p = self.state_derivatives['cake']['cakevolume']

        viscosity_in = self.states['cake']['viscosity']

        c = self.states['bulk']['c']
        c_dot = self.state_derivatives['bulk']['c']

        V = self.states['bulk']['Volume']
        V_dot = self.state_derivatives['bulk']['Volume']

        deltap = self.states['cake']['pressure']

        #parameters
        efficency = self.parameters['efficency']
        molar_volume = self.parameters['molar_volume']
        membrane_area = self.parameters['membrane_area']
        membrane_resistance = self.parameters['membrane_resistance']
        specific_cake_resistance = self.parameters['specific_cake_resistance']

        # Handle inlet DOFs, which are simply copied to the residual
        self.residuals['cake']['c'] = c_in
        self.residuals['cake']['cakevolume'] = calculate_residual_cake_vol_def(Q_in, efficency, molar_volume, c_in, V_dot_C)
        self.residuals['cake']['pressure'] = calculate_residual_press_easy_def(Q_p, V_C, deltap, membrane_area, viscosity_in, membrane_resistance, specific_cake_resistance)
        self.residuals['cake']['permeate'] = calculate_residual_perm_easy_def(Q_in, V_dot_C, Q_p)
        self.residuals['cake']['viscosity'] = calculate_residual_visc_def()

        self.residuals['bulk']['c'] = calculate_residual_concentration_cstr(c, c_dot, V, V_dot, Q_p, Q_out, c_in) 
        self.residuals['bulk']['Volume'] = calculate_residual_volume_cstr(V, V_dot, Q_p, Q_out)
        self.residuals['bulk']['viscosity'] = calculate_residual_visc_cstr()



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
    _state_structures = ['retentate', 'permeate']

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
    _state_structures = ['bulk', 'particle', 'flux']

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