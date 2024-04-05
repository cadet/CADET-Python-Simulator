from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Any, NoReturn

from addict import Dict as ADict
import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import (
    Typed, String, UnsignedInteger, UnsignedFloat,
    SizedNdArray, SizedUnsignedNdArray, NdPolynomial
)
from CADETProcess.dynamicEvents import Section

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
    initial_state : np.ndarray
        Initial state of unit operation
    initial_state_dot : np.ndarray
        Initial state derivative of unit operation
    """

    name = String()
    component_system = Typed(ty=ComponentSystem)

    initial_state = SizedNdArray(size='n_dof')
    initial_state_dot = SizedNdArray(size='n_dof')

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

    @property
    @abstractmethod
    def state_structure(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the structure of the unit operation state blocks.

        Provides a dictionary that maps state names to their respective configurations.
        Each state configuration includes:
        - 'dimensions': An integer or tuple indicating the dimensions associated with
        the state.
        - 'structure': A dictionary detailing the components of the state.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A dictionary where each key is a state block name and each value is a
            dictionary describing that state's configuration.
        """
        pass

    @property
    def n_dof_state_blocks(self) -> Dict[str, int]:
        """Dict[str, int]: Number of degrees of freedom per state block."""
        state_dofs = {}
        for state, state_information in self.state_structure.items():
            n_cells_state = np.prod(state_information['dimensions'], dtype=int)
            n_components_state = sum(state_information['structure'].values())
            state_dofs[state] = n_cells_state * n_components_state
        return state_dofs

    @property
    def n_dof(self) -> int:
        """int: Number of degrees of freedom."""
        return sum([n_states for n_states in self.n_dof_state_blocks.values()])

    def split_state_blocks(self, y) -> Dict[str, np.ndarray]:
        """
        Separate state into its individual blocks.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.

        Returns
        -------
        Dict[str, np.ndarray]
            The state of the unit operation.
        """
        split_state = {}
        start_index = 0
        for state, n_dof in self.n_dof_state_blocks.items():
            end_index = start_index + n_dof
            split_state[state] = y[start_index:end_index]
            start_index = end_index

        return split_state

    def split_state(self, y):
        """
        Separate state into its individual components.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.

        Returns
        -------
        Dict[str, np.ndarray]
            The state of the unit operation.
        """
        y_components = {}

        for state, y_block in self.split_state_blocks(y).items():
            y_components[state] = {}

            dimensions = tuple(self.state_structure[state]['dimensions'])
            components = self.state_structure[state]['structure']

            y_block = y_block.reshape((*dimensions, -1))

            start_index = 0
            for state_s, n_s in components.items():
                end_index = start_index + n_s
                y_components[state][state_s] = y_block[..., start_index:end_index]
                start_index = end_index

        return dict(y_components)

    def split_state_ports(
            self,
            y: np.ndarray,
            target: str
            ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return current state at the unit operation inlet or outlet ports.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.
        target : str, {'inlet', 'outlet'}
            Flag to indicate whether if inlet or outlet ports are to be returned.

        Returns
        -------
        Dict[int, Dict[str, np.ndarray]]
            The current state at the unit operation inlet or outlet ports.
        """
        split_state_ports = {}

        port_mapping = self.port_mapping[target]
        split_state = self.split_state(y)

        for port, mapping_info in port_mapping.items():
            split_state_ports[port] = {}
            mapped_state = mapping_info['mapped_state']
            port_index = mapping_info['port_index']

            y_state = split_state[mapped_state]
            for component, n_entries in self.coupling_state_structure.items():
                y_port = y_state[component][0, ...]

                split_state_ports[port_index][component] = y_port

        return split_state_ports

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

    def construct_coupling_state(self, s: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Construct state dict from np.ndarray.

        Parameters
        ----------
        s : np.ndarray
            New coupling state.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with coupling state structure.
        """
        coupling_state = {}
        start_index = 0
        for component, n_entries in self.coupling_state_structure.items():
            end_index = start_index + n_entries
            coupling_state[component] = s[start_index:end_index]
            start_index = end_index

        return coupling_state

    @property
    def inlet_ports(self) -> Dict[str, int]:
        """dict: Number of inlet ports per state."""
        return {}

    @property
    def n_inlet_ports(self):
        """int: Number of inlet ports."""
        return sum(self.inlet_ports.values())

    @property
    def outlet_ports(self) -> Dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {}

    @property
    def outlet_port_mapping(self) -> Dict[int, str]:
        """dict: Mapping of outlet port indices to corresponding state."""
        outlet_port_mapping = {}

        counter = 0
        for mapped_state, n_ports in self.outlet_ports.items():
            for port in range(n_ports):
                outlet_port_mapping[counter] = {}
                outlet_port_mapping[counter]['mapped_state'] = mapped_state
                outlet_port_mapping[counter]['port_index'] = port
                counter += 1

        return outlet_port_mapping

    @property
    def n_outlet_ports(self):
        """int: Number of inlet ports."""
        return sum(self.outlet_ports.values())

    @property
    def port_mapping(self) -> Dict[int, str]:
        """dict: Mapping of port indices to corresponding state entries."""
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

    # def _validate_coupling_structure(self) -> NoReturn:
    #     for state, n_ports in self.inlet_ports:
    #         print(state, n_ports)
    #         # TODO: Read state structure and verify all coupling state entries are present.
    #     for state, n_ports in self.inlet_ports:
    #         print(state, n_ports)
    #         # TODO: Read state structure and verify all coupling state entries are present.

    def set_inlet_state(
            self,
            y: np.ndarray,
            s: np.ndarray,
            port: int,
            ) -> NoReturn:
        """
        Set the state of the unit operation inlet for a given port.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.
        s : np.ndarray
            New state of the unit operation inlet.
        port : int
            Port of the unit operation for which to set the inlet state.
        """
        if self.n_inlet_ports == 0:
            raise Exception(
                "Cannot set inlet state for unit operation without inlet ports."
            )
        if port > self.n_inlet_ports - 1:
            raise ValueError("Port exceeds number of inlet ports.")

        port_mapping_info = self.port_mapping['inlet'][port]
        inlet_port = port_mapping_info['port_index']

        y_port = self.split_state_ports(y, 'inlet')[inlet_port]

        coupling_state = self.construct_coupling_state(s)
        for key, value in coupling_state.items():
            y_port[key][:] = value[:]

    def get_outlet_state(
            self,
            y: np.ndarray,
            port: int,
            ) -> np.ndarray:
        """
        Return the state of the unit operation outlet for a given port.

        Parameters
        ----------
        y : np.ndarray
            Current state of the unit operation.
        port : int
            Port of the unit operation for which to return the outlet state.

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

        port_mapping_info = self.port_mapping['outlet'][port]
        outlet_port = port_mapping_info['port_index']

        y_port = self.split_state_ports(y, 'outlet')[outlet_port]

        return np.concatenate(list(y_port.values()))

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
    def section_dependent_parameters(self) -> List[str]:
        """list: Section depdendent parameters."""
        return self._section_dependent_parameters

    def update_section_dependent_parameters(
            self,
            start: float,
            end: float,
            parameters: Dict[str, float | np.ndarray]
            ) -> NoReturn:
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
            ) -> Dict[str, np.typing.ArrayLike]:
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
    c : NdPolynomial
        Polynomial coefficients for component concentration.
    viscosity : float
        Viscosity of the solvent.
    """

    c = NdPolynomial(size=('n_comp', 4), default=0)
    viscosity = UnsignedFloat()

    _parameters = ['c', 'viscosity']
    _polynomial_parameters = ['c']
    _section_dependent_parameters = ['c']

    @property
    def state_structure(self) -> Dict[str, Dict[str, Any]]:
        """dict: The state structure for the Inlet unit operation."""
        return {
            'outlet': {
                'dimensions': (1,),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
        }

    @property
    def outlet_ports(self) -> Dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {
            'outlet': 1,
        }

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

    @property
    def state_structure(self) -> Dict[str, Dict[str, Any]]:
        """dict: The state structure for the Outlet unit operation."""
        return {
            'inlet': {
                'dimensions': (1,),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
        }

    @property
    def inlet_ports(self) -> Dict[str, int]:
        """dict: Number of inlet ports per state."""
        return {
            'inlet': 1,
        }

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

    Attributes
    ----------
    volume : float
        Viscosity of the solvent.
    """

    volume = UnsignedFloat()
    c = SizedUnsignedNdArray(size='n_comp')

    _parameters = ['c', 'volume']

    @property
    def state_structure(self) -> Dict[str, Dict[str, Any]]:
        """dict: The state structure for the Cstr unit operation."""
        return {
            'inlet': {
                'dimensions': (1,),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
            'bulk': {
                'dimensions': (1,),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                    'V': 1,
                },
            },
        }

    @property
    def inlet_ports(self) -> Dict[str, int]:
        """dict: Number of inlet ports per state."""
        return {
            'inlet': 1,
        }

    @property
    def outlet_ports(self) -> Dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {
            'bulk': 1,
        }

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

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()
    rejection_model = Typed(ty=RejectionBase)
    cake_compressibility_model = Typed(ty=CakeCompressibilityBase)

    _parameters = [
        'membrane_area',
        'membrane_resistance',
    ]

    @property
    def state_structure(self) -> Dict[str, Dict[str, Any]]:
        """dict: The state structure for the DeadEndFiltration unit operation."""
        return {
            'inlet': {
                'dimensions': (1,),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
            'retentate': {
                'dimensions': (1,),
                'structure': {
                    'Rc': 1,
                    'mc': self.n_comp,
                    'Vp': 1,
                },
            },
            'permeate': {
                'dimensions': (1,),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
        }

    @property
    def inlet_ports(self) -> Dict[str, int]:
        """dict: Number of inlet ports per state."""
        return {
            'inlet': 1,
        }

    @property
    def outlet_ports(self) -> Dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {
            'permeate': 1,
        }

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
    membrane_area : float
        Area of the membrane.
    membrane_resistance : float
        Membrane resistance.
    rejection_model : RejectionBase
        Model for size dependent rejection.

    """

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()
    rejection_model = Typed(ty=RejectionBase)

    n_axial = UnsignedInteger(default=10)

    _parameters = [
        'membrane_area',
        'membrane_resistance',
        'n_axial',
    ]

    @property
    def state_structure(self) -> Dict[str, Dict[str, Any]]:
        """dict: The state structure for the Cstr unit operation."""
        return {
            'retentate': {
                'dimensions': (self.n_axial, ),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
            'permeate': {
                'dimensions': (self.n_axial, ),
                'structure': {
                    'c': self.n_comp,
                    'viscosity': 1,
                },
            },
        }

    @property
    def inlet_ports(self) -> Dict[str, int]:
        """dict: Number of inlet ports per state."""
        return {
            'retentate': 1,
        }

    @property
    def outlet_ports(self) -> Dict[str, int]:
        """dict: Number of outlet ports per state."""
        return {
            'retentate': 1,
            'permeate': 1,
        }

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
