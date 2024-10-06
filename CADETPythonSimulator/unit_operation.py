from abc import abstractmethod
from collections import defaultdict
from typing import Any, NoReturn, Optional

import numpy as np
import numpy.typing as npt

from CADETProcess.processModel import ComponentSystem
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Typed, String, UnsignedInteger, UnsignedFloat, SizedUnsignedNdArray, NdPolynomial
)
from CADETProcess.dynamicEvents import Section

from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.exception import NotInitializedError, CADETPythonSimError
from CADETPythonSimulator.state import State, state_factory
from CADETPythonSimulator.residual import (
    calculate_residual_volume_cstr,
    calculate_residual_concentration_cstr,
    calculate_residual_visc_cstr,
    calculate_residual_press_easy_def,
    calculate_residual_cake_vol_def,
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
    component_system = Typed(ty=CPSComponentSystem)

    _state_structures = []
    _parameters = []
    _section_dependent_parameters = []

    def __init__(self, component_system, name, *args, **kwargs):
        """Construct the UnitOperationBase."""
        self.component_system = component_system
        self.name = name

        super().__init__(*args, **kwargs)

        self.parameter_sections = {}

        self._states: Optional[dict[str, State]] = None
        self._state_derivatives: Optional[dict[str, State]] = None
        self._residuals: Optional[dict[str, State]] = None

        self._Q_in: Optional[np.ndarray] = None
        self._Q_out: Optional[np.ndarray] = None
        self._coupling_state_structure = {
            'c': component_system.n_comp
        }

    @property
    def n_dof(self) -> int:
        """int: Total number of degrees of freedom."""
        return sum([state.n_dof for state in self.states.values()])

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self.component_system.n_comp

    def initialize_state(self) -> NoReturn:
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
        """dict: State array block of the unit operation, indexed by name."""
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
        """dict: State derivative array block of the unit operation, indexed by name."""
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
    def y_init(self) -> np.ndarray:
        """np.ndarray: State derivative array flattened into one dimension."""
        return np.concatenate([
                state_derivative.s_flat
                for state_derivative in self.state_derivatives.values()
        ])

    @y_init.setter
    def y_init(self, y_init: np.ndarray) -> NoReturn:
        start_index = 0
        for state_derivative in self.state_derivatives.values():
            end_index = start_index + state_derivative.n_dof
            state_derivative.s_flat = y_init[start_index:end_index]
            start_index = end_index

    @property
    def residuals(self) -> dict[str, State]:
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

    @Q_in.setter
    def Q_in(self, Q_in: npt.ArrayLike):
        if not (len(Q_in) == self.n_inlet_ports):
            raise ValueError(
                f"""Q_in has wrong shape. Expected ({self.n_inlet_ports},),
                got ({len(Q_in)},)"""
            )
        self._Q_in = np.array(Q_in)

    @property
    def Q_out(self) -> np.ndarray:
        """np.ndarray: Ingoing flow rates."""
        if self._Q_out is None:
            raise NotInitializedError(
                "Unit operation flow rates are not yet initialized."
            )

        return self._Q_out

    @Q_out.setter
    def Q_out(self, Q_out: npt.ArrayLike):
        if not (len(Q_out) == self.n_outlet_ports):
            raise ValueError(
                f"""Q_out has wrong shape. Expected ({self.n_outlet_ports},),
                got ({len(Q_out)},)"""
            )
        self._Q_out = np.array(Q_out)

    def set_Q_in_port(self, port: int, Q_in: float):
        """
        Set a portspecific Q_in.

        Parameters
        ----------
        port : int
            port to set Q
        Q_in : float
            rate to set

        """
        if not port < self.n_inlet_ports and port >= 0:
            raise f"""Port {port} is not inbetween 0 and {self.n_inlet_ports}"""
        self._Q_in[port] = Q_in

    def set_Q_out_port(self, port: int, Q_out: float):
        """
        Set a portspecific Q_out.

        Parameters
        ----------
        port : int
            port to set Q
        Q_out : float
            rate to set

        """
        if not port < self.n_outlet_ports and port >= 0:
            raise f"""Port {port} is not inbetween 0 and {self.n_outlet_ports}"""
        self._Q_out[port] = Q_out

    @property
    def coupling_state_structure(self) -> dict:
        """dict: State structure that must be accessible in inlet / outlet ports."""
        return self._coupling_state_structure

    @coupling_state_structure.setter
    def coupling_state_structure(self, coupling_state_structure: dict) -> NoReturn:
        """Set coupling state structure."""
        self._coupling_state_structure = coupling_state_structure

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
            ) -> dict[str, np.ndarray]:
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
            ) -> dict[str, np.ndarray]:
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
            ) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        pass

    @property
    def section_dependent_parameters(self) -> list[str]:
        """list: Section depdendent parameters."""
        return self._section_dependent_parameters

    def update_parameters(
            self,
            start: float,
            end: float,
            parameters: dict[str, float | np.ndarray]
            ) -> NoReturn:
        """
        Update parameters.

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
        for param, value in parameters.items():
            is_polynomial = param in self.polynomial_parameters
            setattr(self, param, value)
            if param in self.section_dependent_parameters:
                coeffs = getattr(self, param)
                self.parameter_sections[param] =\
                    Section(start, end, coeffs, is_polynomial)

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

    def initialize_initial_values(self, t_zero: float):
        """
        Initialize Derivatives if possible.

        Initializes with zeroes by default.

        Parameters
        ----------
        t_zero: float
            initial value at time t_zero

        """
        self.y_dot = np.zeros(self.n_dof)

    def __str__(self) -> str:
        """Return string represenation of the unit operation."""
        return self.name

    def compute_residual_for_initial_values(self, t_zero: float):
        """Calculate Residual for initial values."""
        self.compute_residual(t_zero)


class Inlet(UnitOperationBase):
    """
    System inlet.

    Attributes
    ----------
    c_poly : NdPolynomial
        Polynomial coefficients for component concentration.

    """

    c_poly = NdPolynomial(size=('n_comp', 4), default=0)
    viscosity = UnsignedFloat()

    _parameters = ['c_poly']
    _polynomial_parameters = ['c_poly']
    _section_dependent_parameters = ['c_poly']

    outlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp'},
        'n_outlet_ports': 1,
    }
    _state_structures = ['outlet']

    def compute_residual(
            self,
            t: float
            ) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        c = self.states['outlet']['c']
        t_poly = np.array([1, t, t**2, t**3])
        self.residuals['outlet']['c'] = self.c_poly @ t_poly - c

    def initialize_initial_values(self, t_zero: float):
        """
        Initialize initial values for Inlet Unit Operation.

        Parameters
        ----------
        t_zero : float
            Time to initialize the values

        """
        t_poly = np.array([1, t_zero, t_zero**2, t_zero**3])
        self.states['outlet']['c'] = self.c_poly @ t_poly
        t_poly = np.array([0, 1, 2*t_zero, 3*t_zero**2])
        self.state_derivatives['outlet']['c'] = self.c_poly @ t_poly

class Outlet(UnitOperationBase):
    """System outlet."""

    inlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp'},
        'n_inlet_ports': 1,
    }
    _state_structures = ['inlet']

    def compute_residual(
            self,
            t: float
            ) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        self.residuals['inlet']['c'] -= self.states['inlet']['c']


class Cstr(UnitOperationBase):
    """Continuous stirred tank reactor."""

    inlet = {
        'dimensions': (),
        'entries': {'c': 'n_comp'},
        'n_inlet_ports': 1,
    }
    bulk = {
        'dimensions': (),
        'entries': {'c': 'n_comp', 'Volume': 1},
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
        # c_in_dot = self.state_derivatives['inlet']['c']

        c = self.states['bulk']['c']
        c_dot = self.state_derivatives['bulk']['c']

        V = self.states['bulk']['Volume']
        V_dot = self.state_derivatives['bulk']['Volume']

        # Handle inlet DOFs, which are simply copied to the residual
        self.residuals['inlet']['c'] -= c_in

        # Handle bulk/outlet DOFs
        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]

        self.residuals['bulk']['c'] = calculate_residual_concentration_cstr(
            c, c_dot, V, V_dot, Q_in, Q_out, c_in
        )

        self.residuals['bulk']['Volume'] = calculate_residual_volume_cstr(
            V, V_dot, Q_in, Q_out
        )

    def initialize_initial_values(self, t_zero: float):
        """
        Initialize initial values for Inlet Unit Operation.

        Parameters
        ----------
        t_zero : float
            Time to initialize the values

        """
        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]
        V = self.states['bulk']['Volume']
        V_dot = Q_in - Q_out
        c_in = self.states['inlet']['c']
        c = self.states['bulk']['c']

        self.state_derivatives['bulk']['c'] = - V_dot/V*c + Q_in/V*c_in - Q_out/V*c
        self.state_derivatives['bulk']['volume'] = V_dot
        self.state_derivatives['inlet']['c'] = np.zeros(self.n_comp)

class DeadEndFiltration(UnitOperationBase):
    """
    Dead end filtration model.

    Attributes
    ----------
    membrane_area : float
        Area of the membrane.
    membrane_resistance : float
        Membrane resistance.
    solution_viscosity : float
        viscosity of the solution
    rejection_model : RejectionBase
        Model for size dependent rejection.
    cake_compressibility_model : CakeCompressibilityBase
        Model for cake compressibility.
    viscosity_model : ViscosityBase
        Model for viscosities

    """

    cake = {
        'dimensions': (),
        'entries': {
            'c': 'n_comp',
            'n_feed': 'n_comp',
            'cakevolume': 'n_comp',
            'n_cake': 'n_comp',
            'permeatevolume': 1,
            'n_permeate': 'n_comp',
            'c_permeate': 'n_comp',
            'pressure': 1
        },
        'n_inlet_ports': 1,
    }
    permeate_tank = {
        'dimensions': (),
        'entries': {
            'c': 'n_comp',
            'tankvolume': 1
        },
        'n_outlet_ports': 1,
    }

    _state_structures = ['cake', 'permeate_tank']

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()
    rejection_model = Typed(ty=RejectionBase)
    viscosity_model = Typed(ty=ViscosityBase)

    _parameters = [
        'membrane_area',
        'membrane_resistance',
        'rejection_model',
        'viscosity_model'
    ]

    def compute_residual(
            self,
            t: float,
        ) -> NoReturn:
        """Calculate the Residuum for DEF."""
        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]

        c_feed = self.states['cake']['c']
        c_feed_dot = self.state_derivatives['cake']['c']

        n_feed = self.states['cake']['n_feed']
        n_feed_dot = self.state_derivatives['cake']['n_feed']

        n_cake = self.states['cake']['n_cake']
        n_cake_dot = self.state_derivatives['cake']['n_cake']

        cake_vol = self.states['cake']['cakevolume']
        cake_vol_dot = self.state_derivatives['cake']['cakevolume']

        c_permeate = self.states['cake']['c_permeate']
        c_permeate_dot = self.state_derivatives['cake']['c_permeate']

        n_permeate = self.states['cake']['n_permeate']
        n_permeate_dot = self.state_derivatives['cake']['n_permeate']

        permeate_vol = self.states['cake']['permeatevolume']
        permeate_vol_dot = self.state_derivatives['cake']['permeatevolume']

        deltap = self.states['cake']['pressure']

        c_tank = self.states['permeate_tank']['c']
        c_tank_dot = self.state_derivatives['permeate_tank']['c']

        tankvolume = self.states['permeate_tank']['tankvolume']
        tankvolume_dot = self.state_derivatives['permeate_tank']['tankvolume']

        # parameters
        molecular_weights = np.array(self.component_system.molecular_weights)
        densities = np.array(self.component_system.pure_densities)
        viscosities = np.array(self.component_system.viscosities)
        membrane_area = self.parameters['membrane_area']
        membrane_resistance = self.parameters['membrane_resistance']
        specific_cake_resistance =\
            np.array(self.component_system.specific_cake_resistances)

        rejection = np.array(
                        [
                            self.rejection_model.get_rejection(mw)\
                            for mw in molecular_weights
                        ]
                    )

        # Coupling residual equation
        self.residuals['cake']['c'] -= c_feed

        # Number of Feed

        self.residuals['cake']['n_feed'] = n_feed_dot - Q_in * c_feed

        # Number of cake

        self.residuals['cake']['n_cake'] = n_cake_dot - rejection * n_feed_dot

        # Number of Permeate

        self.residuals['cake']['n_permeate'] =\
            n_permeate_dot - (1 - rejection) * n_feed_dot

        # Cakevolume

        self.residuals['cake']['cakevolume'] =\
            cake_vol_dot - n_cake_dot * molecular_weights / densities

        # Permeate flow

        self.residuals['cake']['permeatevolume'] =\
            permeate_vol_dot - np.sum(n_permeate_dot * molecular_weights / densities)

        # Concentration Permeate

        self.residuals['cake']['c_permeate'] =\
            c_permeate - n_permeate_dot / permeate_vol_dot

        # Pressure equation

        cakresistance = \
            np.sum(specific_cake_resistance * densities * cake_vol/membrane_area)

        viscositiy = \
            np.exp(np.sum(n_permeate_dot* np.log(viscosities)) / np.sum(n_permeate_dot))

        self.residuals['cake']['pressure'] = \
            viscositiy * permeate_vol_dot * (membrane_resistance + cakresistance)\
            / membrane_area - deltap

        # Tank equations

        self.residuals['permeate_tank']['c'] = calculate_residual_concentration_cstr(
            c=c_tank,
            c_dot=c_tank_dot,
            V=tankvolume,
            V_dot=tankvolume_dot,
            Q_in=permeate_vol_dot,
            Q_out=Q_out,
            c_in=c_permeate
        )

        self.residuals['permeate_tank']['tankvolume'] =\
            tankvolume_dot - permeate_vol_dot + Q_out


    @property
    def y_init(self) -> np.ndarray:
        """np.ndarray: State derivative array flattened into one dimension."""
        ret = []
        for state, state_derivative in self.state_derivatives.items():
            if state != "cake":
                ret.append(state_derivative.s_flat)
            else:
                for entry in state_derivative.entries.keys():
                    if entry != "pressure":
                        ret.append(state_derivative[entry])
                    else:
                        ret.append(self.states[state][entry])
        return np.concatenate(ret, axis=None)

    @y_init.setter
    def y_init(self, y_init: np.ndarray) -> NoReturn:
        start_index = 0
        for state, state_derivative in self.state_derivatives.items():
            end_index = start_index + state_derivative.n_dof
            if state != "cake":
                state_derivative.s_flat = y_init[start_index:end_index]
            else:
                sub_start_index = start_index
                for entry, dim in state_derivative.entries.items():
                    sub_end_index = sub_start_index + dim
                    if state != "pressure":
                        self.state_derivatives[state][entry] =\
                            y_init[sub_start_index:sub_end_index]
                    else:
                        self.states[state][entry] =\
                            y_init[sub_start_index:sub_end_index]
                    sub_start_index = sub_end_index
            start_index = end_index

    def initialize_initial_values(self, t_zero: float):
        """Initialize the values."""
        molecular_weights = np.array(self.component_system.molecular_weights)
        densities = np.array(self.component_system.pure_densities)
        viscosities = np.array(self.component_system.viscosities)
        membrane_area = self.parameters['membrane_area']
        membrane_resistance = self.parameters['membrane_resistance']
        specific_cake_resistance =\
            np.array(self.component_system.specific_cake_resistances)

        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]
        c_feed = self.states['cake']['c']

        n_feed_dot = Q_in * c_feed
        self.state_derivatives['cake']['n_feed'] = n_feed_dot

        rejection = np.array(
                [
                    self.rejection_model.get_rejection(mw)\
                    for mw in molecular_weights
                ]
            )

        n_cake_dot = rejection * n_feed_dot
        self.state_derivatives['cake']['n_cake'] = n_cake_dot

        cake_vol = self.states['cake']['cakevolume']

        cake_vol_dot = molecular_weights * n_cake_dot / densities

        self.state_derivatives['cake']['cakevolume'] = cake_vol_dot


        n_permeate_dot = (1 - rejection) * n_feed_dot
        self.state_derivatives['cake']['n_permeate'] = n_permeate_dot

        c_permeate_dot = self.state_derivatives['cake']['c_permeate']

        permeate_vol_dot = np.sum(n_permeate_dot * molecular_weights / densities)
        self.state_derivatives['cake']['permeatevolume'] = permeate_vol_dot

        c_permeate = n_permeate_dot / permeate_vol_dot
        self.states['cake']['c_permeate'] = c_permeate


        cakresistance = \
            np.sum(specific_cake_resistance * densities * cake_vol/membrane_area)

        viscositiy = \
            np.exp(np.sum(n_permeate_dot* np.log(viscosities)) / np.sum(n_permeate_dot))

        self.states['cake']['pressure'] = \
            viscositiy * permeate_vol_dot * (membrane_resistance + cakresistance)\
            / membrane_area

        c_tank = self.states['permeate_tank']['c']

        tankvolume = self.states['permeate_tank']['tankvolume']
        tankvolume_dot = permeate_vol_dot - Q_out
        self.state_derivatives['permeate_tank']['tankvolume'] = tankvolume_dot

        if tankvolume == 0:
            raise CADETPythonSimError("""Initialize error
                        Volume of Permeate tank can't be initialized with 0""")
        c_tank_dot =\
            (permeate_vol_dot * c_permeate - Q_out * c_tank -c_tank*tankvolume_dot)\
            /tankvolume
        self.state_derivatives['permeate_tank']['c'] = c_tank_dot



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
    rejection : RejectionBase
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

    def __init__(self, component_system, name, *args, **kwargs):
        """Construct CFF."""
        super().__init__(component_system, name, *args, **kwargs)
        self.coupling_state_structure['viscosity'] = 1

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
