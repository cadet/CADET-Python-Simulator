from abc import abstractmethod
from typing import NoReturn, Optional

import numpy as np
import numpy.typing as npt
from CADETProcess.dataStructure import (
    NdPolynomial,
    String,
    Structure,
    Typed,
    UnsignedFloat,
    UnsignedInteger,
)
from CADETProcess.dynamicEvents import Section

from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.distribution_base import DistributionBase
from CADETPythonSimulator.exception import NotInitializedError
from CADETPythonSimulator.geometry import (
    Cylinder,
    DiscretizedGeometry,
    RadialAngularSymmetry,
    WellMixed,
)
from CADETPythonSimulator.rejection import RejectionBase
from CADETPythonSimulator.residual import (
    calculate_residual_concentration_cstr,
    calculate_residual_volume_cstr,
)
from CADETPythonSimulator.state import State, state_factory
from CADETPythonSimulator.viscosity import ViscosityBase


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

    # Named ports exposed to the System. Keys are port names (used in connect()),
    # values are state names (which state's boundary cells are read/written).
    _inlet_ports: dict[str, str] = {}
    _outlet_ports: dict[str, str] = {}

    def _make_state_geometry(self, state_name: str) -> DiscretizedGeometry:
        """Return the DiscretizedGeometry for a given state block. Defaults to 0D."""
        return DiscretizedGeometry(Cylinder(length=1.0), WellMixed())

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
            geometry = self._make_state_geometry(state_block)

            state = state_factory(
                self, state_block, geometry=geometry, **state_structure
            )
            self._states[state.name] = state

            state_derivative = state_factory(
                self, state_block, geometry=geometry, **state_structure
            )
            self._state_derivatives[state_derivative.name] = state_derivative

            residual = state_factory(
                self, state_block, geometry=geometry, **state_structure
            )
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
        return np.concatenate(
            [
                state_derivative.s_flat
                for state_derivative in self.state_derivatives.values()
            ]
        )

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
        return np.concatenate(
            [
                state_derivative.s_flat
                for state_derivative in self.state_derivatives.values()
            ]
        )

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
        return np.concatenate([residual.s_flat for residual in self.residuals.values()])

    @r.setter
    def r(self, r: np.ndarray) -> NoReturn:
        start_index = 0
        for residual in self.residuals.values():
            end_index = start_index + residual.n_dof
            residual.s_flat = r[start_index:end_index]
            start_index = end_index

    @property
    def Q_in(self) -> np.ndarray:
        """np.ndarray: Ingoing flow rates, indexed by _inlet_ports declaration order."""
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
        """np.ndarray: Outgoing flow rates, indexed by _outlet_ports declaration order."""
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

    @property
    def n_inlet_ports(self) -> int:
        """int: Number of inlet ports."""
        return len(self._inlet_ports)

    @property
    def n_outlet_ports(self) -> int:
        """int: Number of outlet ports."""
        return len(self._outlet_ports)

    def get_port_state(self, port_name: str) -> dict[str, np.ndarray]:
        """Return entry arrays at the named outlet port."""
        try:
            state_name = self._outlet_ports[port_name]
        except KeyError:
            raise ValueError(f"'{port_name}' is not an outlet port of '{self.name}'.")
        return self.states[state_name].get_port_state("outlet")

    def set_port_state(self, port_name: str, values: dict[str, np.ndarray]) -> None:
        """Set entry values at the named inlet port (in-place)."""
        try:
            state_name = self._inlet_ports[port_name]
        except KeyError:
            raise ValueError(f"'{port_name}' is not an inlet port of '{self.name}'.")
        self.states[state_name].set_port_state("inlet", values)

    @abstractmethod
    def compute_residual(self, t: float) -> NoReturn:
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
        self, start: float, end: float, parameters: dict[str, float | np.ndarray]
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
                self.parameter_sections[param] = Section(
                    start, end, coeffs, is_polynomial
                )

    def get_parameter_values_at_time(self, t: float) -> dict[str, np.typing.ArrayLike]:
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

    c_poly = NdPolynomial(size=("n_comp", 4), default=0)
    viscosity = UnsignedFloat()

    _parameters = ["c_poly"]
    _polynomial_parameters = ["c_poly"]
    _section_dependent_parameters = ["c_poly"]

    _inlet_ports = {}
    _outlet_ports = {"outlet": "outlet"}

    outlet = {"entries": {"c": "n_comp"}}
    _state_structures = ["outlet"]

    def compute_residual(self, t: float) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        t_K = t
        c = self.states["outlet"]["c"]
        t_poly = np.array([1, t_K, t_K**2, t_K**3])
        self.residuals["outlet"]["c"] = self.c_poly @ t_poly - c

    def initialize_initial_values(self, t_zero: float):
        """
        Initialize initial values for Inlet Unit Operation.

        Parameters
        ----------
        t_zero : float
            Time to initialize the values

        """
        t_zero = 0
        t_poly = np.array([1, t_zero, t_zero**2, t_zero**3])
        self.states["outlet"]["c"] = self.c_poly @ t_poly
        t_poly = np.array([0, 1, 2 * t_zero, 3 * t_zero**2])
        self.state_derivatives["outlet"]["c"] = self.c_poly @ t_poly


class DistributionInlet(UnitOperationBase):
    """
    System inlet.

    Attributes
    ----------
    distribution_function : DistributionBase
        Function to describe a Volume profile.

    """

    distribution_function = Typed(ty=DistributionBase)
    section_number = Typed(ty=int)

    _inlet_ports = {}
    _outlet_ports = {"outlet": "outlet"}

    outlet = {"entries": {"c": "n_comp"}}
    _state_structures = ["outlet"]
    _parameters = ["section_number"]
    _section_dependent_parameters = ["section_number"]

    def compute_residual(self, t: float) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        c = self.states["outlet"]["c"]
        nr = self.section_number
        xi = self.distribution_function.get_distribution(t, nr)
        molecular_weights = np.array(self.component_system.molecular_weights)
        densities = np.array(self.component_system.densities)

        ci = xi * densities / molecular_weights

        self.residuals["outlet"]["c"] = ci - c

    def initialize_initial_values(self, t_zero: float):
        """
        Initialize initial values for Inlet Unit Operation.

        Parameters
        ----------
        t_zero : float
            Time to initialize the values

        """
        nr = self.parameters["section_number"]
        xi = self.distribution_function.get_distribution(t_zero, nr)
        molecular_weights = np.array(self.component_system.molecular_weights)
        densities = np.array(self.component_system.densities)

        ci = xi * densities / molecular_weights

        self.states["outlet"]["c"] = ci


class ConcentrationDistributionInlet(UnitOperationBase):
    """
    System inlet.

    Attributes
    ----------
    distribution_function : DistributionBase
        Function to describe concentration profile.

    """

    distribution_function = Typed(ty=DistributionBase)
    section_number = Typed(ty=int)

    _inlet_ports = {}
    _outlet_ports = {"outlet": "outlet"}

    outlet = {"entries": {"c": "n_comp"}}
    _state_structures = ["outlet"]
    _parameters = ["section_number"]
    _section_dependent_parameters = ["section_number"]

    def compute_residual(self, t: float) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        c = self.states["outlet"]["c"]
        nr = self.section_number
        ci = self.distribution_function.get_distribution(t, nr)

        self.residuals["outlet"]["c"] = ci - c

    def initialize_initial_values(self, t_zero: float):
        """
        Initialize initial values for Inlet Unit Operation.

        Parameters
        ----------
        t_zero : float
            Time to initialize the values

        """
        nr = self.parameters["section_number"]
        ci = self.distribution_function.get_distribution(t_zero, nr)

        self.states["outlet"]["c"] = ci


class Outlet(UnitOperationBase):
    """System outlet."""

    _inlet_ports = {"inlet": "inlet"}
    _outlet_ports = {}

    inlet = {"entries": {"c": "n_comp"}}
    _state_structures = ["inlet"]

    def compute_residual(self, t: float) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        self.residuals["inlet"]["c"] -= self.states["inlet"]["c"]


class Cstr(UnitOperationBase):
    """Continuous stirred tank reactor."""

    _inlet_ports = {"inlet": "inlet"}
    _outlet_ports = {"outlet": "bulk"}

    inlet = {"entries": {"c": "n_comp"}}
    bulk = {"entries": {"c": "n_comp", "Volume": 1}}
    _state_structures = ["inlet", "bulk"]

    def compute_residual(self, t: float) -> NoReturn:
        """
        Calculate the residual of the unit operation at time `t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the residual.

        """
        c_in = self.states["inlet"]["c"]

        c = self.states["bulk"]["c"]
        c_dot = self.state_derivatives["bulk"]["c"]

        V = self.states["bulk"]["Volume"]
        V_dot = self.state_derivatives["bulk"]["Volume"]

        # Handle inlet DOFs, which are simply copied to the residual
        self.residuals["inlet"]["c"] -= c_in

        # Handle bulk/outlet DOFs
        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]

        self.residuals["bulk"]["c"] = calculate_residual_concentration_cstr(
            c, c_dot, V, V_dot, Q_in, Q_out, c_in
        )

        self.residuals["bulk"]["Volume"] = calculate_residual_volume_cstr(
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
        V = self.states["bulk"]["Volume"]
        V_dot = Q_in - Q_out
        c_in = self.states["inlet"]["c"]
        c = self.states["bulk"]["c"]

        self.state_derivatives["bulk"]["c"] = (
            -V_dot / V * c + Q_in / V * c_in - Q_out / V * c
        )
        self.state_derivatives["bulk"]["volume"] = V_dot
        self.state_derivatives["inlet"]["c"] = np.zeros(self.n_comp)


class DeadEndFiltration(UnitOperationBase):
    """
    Dead end filtration model.

    Attributes
    ----------
    membrane_area : float
        Area of the membrane. [m^2]
    membrane_resistance : float
        Membrane resistance. [m^{-1}]
    rejection_model : RejectionBase
        Model for size dependent rejection.
    viscosity_model : ViscosityBase
        Model for viscosities

    """

    _inlet_ports = {"inlet": "inlet"}
    _outlet_ports = {"outlet": "permeate_tank"}

    inlet = {"entries": {"c": "n_comp", "n": "n_comp"}}
    permeate = {"entries": {"c": "n_comp", "n": "n_comp", "V": 1}}
    retentate = {"entries": {"c": "n_comp", "n": "n_comp", "V": 1}}
    cake = {"entries": {"c": "n_comp", "n": "n_comp", "V": 1, "pressure": 1}}
    permeate_tank = {"entries": {"c": "n_comp", "n": "n_comp", "V": 1}}

    _state_structures = ["inlet", "permeate", "retentate", "cake", "permeate_tank"]

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()
    rejection_model = Typed(ty=RejectionBase)
    viscosity_model = Typed(ty=ViscosityBase)

    _parameters = [
        "membrane_area",
        "membrane_resistance",
        "rejection_model",
        "viscosity_model",
    ]

    def compute_residual(self, t: float) -> NoReturn:
        """Calculate the Residuum for DEF."""
        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]

        c_in = self.states["inlet"]["c"]
        n_in_dot = self.state_derivatives["inlet"]["n"]

        c_R = self.states["retentate"]["c"]
        n_R_dot = self.state_derivatives["retentate"]["n"]
        Q_R = self.state_derivatives["retentate"]["V"]

        c_P = self.states["permeate"]["c"]
        n_P_dot = self.state_derivatives["permeate"]["n"]
        Q_P = self.state_derivatives["permeate"]["V"]

        vol_cake = self.states["cake"]["V"]
        vol_cake_dot = self.state_derivatives["cake"]["V"]
        n_C_dot = self.state_derivatives["cake"]["n"]
        n_C = self.states["cake"]["n"]
        c_C = self.states["cake"]["c"]
        c_C_dot = self.state_derivatives["cake"]["c"]
        delta_p = self.states["cake"]["pressure"]

        c_PT = self.states["permeate_tank"]["c"]
        c_PT_dot = self.state_derivatives["permeate_tank"]["c"]
        n_PT = self.states["permeate_tank"]["n"]
        vol_PT = self.states["permeate_tank"]["V"]
        vol_PT_dot = self.state_derivatives["permeate_tank"]["V"]

        # parameters
        molecular_weights = np.array(self.component_system.molecular_weights)
        densities = np.array(self.component_system.densities)
        viscosities = np.array(self.component_system.viscosities)
        membrane_area = self.parameters["membrane_area"]
        membrane_resistance = self.parameters["membrane_resistance"]
        specific_cake_resistance = np.array(
            self.component_system.specific_cake_resistances
        )

        rejection = np.array(
            [self.rejection_model.get_rejection(mw) for mw in molecular_weights]
        )

        # Inlet Equations
        self.residuals["inlet"]["c"] -= c_in
        self.residuals["inlet"]["n"] = n_in_dot - Q_in * c_in

        # Equations for permeate and retentate

        self.residuals["retentate"]["n"] = n_R_dot - rejection * n_in_dot
        self.residuals["permeate"]["n"] = n_P_dot - (1 - rejection) * n_in_dot

        self.residuals["retentate"]["V"] = Q_R - np.sum(
            n_R_dot * molecular_weights / densities
        )
        self.residuals["permeate"]["V"] = Q_P - np.sum(
            n_P_dot * molecular_weights / densities
        )

        if Q_R > 1e-16:
            self.residuals["retentate"]["c"] = c_R - n_R_dot / Q_R
        else:
            self.residuals["retentate"]["c"] = c_R

        if Q_P > 1e-16:
            self.residuals["permeate"]["c"] = c_P - n_P_dot / Q_P
        else:
            self.residuals["permeate"]["c"] = c_P

        # Cake equations

        self.residuals["cake"]["V"] = vol_cake - np.sum(
            n_C * molecular_weights / densities
        )
        self.residuals["cake"]["n"] = n_C_dot - n_R_dot

        if vol_cake > 1e-16:
            self.residuals["cake"]["c"] = (
                c_C_dot - (n_C_dot - c_C * vol_cake_dot) / vol_cake
            )
        else:
            self.residuals["cake"]["c"] = c_C_dot

        # Permeate tank equations

        c_tank_dot_new = (n_P_dot - Q_out * c_PT - vol_PT_dot * c_PT) / vol_PT

        self.residuals["permeate_tank"]["c"] = c_PT_dot - c_tank_dot_new

        self.residuals["permeate_tank"]["V"] = vol_PT_dot - Q_P + Q_out

        self.residuals["permeate_tank"]["n"] = n_PT - c_PT * vol_PT

        # Pressure equation

        if not np.sum(n_P_dot) < 1e-16:
            fractions = n_P_dot / sum(n_P_dot)

            viscosity = self.viscosity_model.get_mixture_viscosity(
                viscosities, fractions
            )

            component_volume = n_C * molecular_weights / densities
            cake_resistance = np.sum(
                specific_cake_resistance * densities * component_volume / membrane_area
            )

            self.residuals["cake"]["pressure"] = (
                delta_p
                - viscosity
                * Q_P
                * (membrane_resistance + cake_resistance)
                / membrane_area
            )
        else:
            self.residuals["cake"]["pressure"] = delta_p

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
                    if entry != "pressure":
                        self.state_derivatives[state][entry] = y_init[
                            sub_start_index:sub_end_index
                        ]
                    else:
                        self.states[state][entry] = y_init[
                            sub_start_index:sub_end_index
                        ]
                    sub_start_index = sub_end_index
            start_index = end_index

    def initialize_initial_values(self, t_zero: float):
        """Initialize the values."""
        molecular_weights = np.array(self.component_system.molecular_weights)
        densities = np.array(self.component_system.densities)
        viscosities = np.array(self.component_system.viscosities)
        membrane_area = self.parameters["membrane_area"]
        membrane_resistance = self.parameters["membrane_resistance"]
        specific_cake_resistance = np.array(
            self.component_system.specific_cake_resistances
        )

        Q_in = self.Q_in[0]
        Q_out = self.Q_out[0]
        c_in = self.states["inlet"]["c"]

        n_in_dot = Q_in * c_in
        self.state_derivatives["inlet"]["n"] = n_in_dot

        rejection = np.array(
            [self.rejection_model.get_rejection(mw) for mw in molecular_weights]
        )

        n_cake_dot = rejection * n_in_dot
        self.state_derivatives["retentate"]["n"] = n_cake_dot
        self.state_derivatives["cake"]["n"] = n_cake_dot

        vol_cake_dot = np.sum(molecular_weights * n_cake_dot / densities)
        self.state_derivatives["cake"]["V"] = vol_cake_dot
        self.state_derivatives["retentate"]["V"] = vol_cake_dot

        if vol_cake_dot > 1e-16:
            self.states["retentate"]["c"] = n_cake_dot / vol_cake_dot
        else:
            self.states["retentate"]["c"][:] = 0.0

        vol_cake = self.states["cake"]["V"]
        n_cake = self.states["cake"]["n"]

        if vol_cake > 1e-16:
            self.states["cake"]["c"] = n_cake
        else:
            self.states["cake"]["c"][:] = 0

        n_P_dot = (1 - rejection) * n_in_dot
        self.state_derivatives["permeate"]["n"] = n_P_dot
        Q_P = np.sum(n_P_dot * molecular_weights / densities)

        self.state_derivatives["permeate"]["V"] = Q_P

        if Q_P > 1e-16:
            self.states["permeate"]["c"] = n_P_dot / Q_P
        else:
            self.states["permeate"]["c"][:] = 0.0

        c_PT = self.states["permeate_tank"]["c"]
        V_PT = self.states["permeate_tank"]["V"]

        self.states["permeate_tank"]["n"] = c_PT * V_PT

        V_PT_dot = Q_P - Q_out

        c_PT_dot = (n_P_dot - Q_out * c_PT - V_PT_dot * c_PT) / V_PT

        self.state_derivatives["permeate_tank"]["c"] = c_PT_dot

        self.state_derivatives["permeate_tank"]["V"] = V_PT_dot

        if not np.sum(n_P_dot) < 1e-16:
            fractions = n_P_dot / sum(n_P_dot)

            viscosity = self.viscosity_model.get_mixture_viscosity(
                viscosities, fractions
            )

            n_C = self.states["cake"]["n"]
            component_volume = n_C * molecular_weights / densities
            cake_resistance = np.sum(
                specific_cake_resistance * densities * component_volume / membrane_area
            )

            self.states["cake"]["pressure"] = (
                viscosity
                * Q_P
                * (membrane_resistance + cake_resistance)
                / membrane_area
            )
        else:
            self.states["cake"]["pressure"] = 0


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

    _inlet_ports = {"inlet": "retentate"}
    _outlet_ports = {"retentate": "retentate", "permeate": "permeate"}

    retentate = {"entries": {"c": "n_comp", "viscosity": 1, "Volume": 1}}
    permeate = {"entries": {"c": "n_comp", "viscosity": 1, "Volume": 1}}
    _state_structures = ["retentate", "permeate"]

    def _make_state_geometry(self, state_name: str) -> DiscretizedGeometry:
        return DiscretizedGeometry(
            Cylinder(length=1.0), RadialAngularSymmetry(), {"axial": self.n_axial}
        )

    rejection_model = Typed(ty=RejectionBase)

    membrane_area = UnsignedFloat()
    membrane_resistance = UnsignedFloat()

    _parameters = ["membrane_area", "membrane_resistance"]

    def compute_residual(
        self, t: float, y: np.ndarray, y_dot: np.ndarray, residual: np.ndarray
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

    _inlet_ports = {"inlet": "bulk"}
    _outlet_ports = {"outlet": "bulk"}

    bulk = {"entries": {"c": "n_comp", "viscosity": 1}}
    particle = {"entries": {"c": "n_comp", "viscosity": 1, "q": "n_comp"}}
    flux = {"entries": {"c": "n_comp"}}
    _state_structures = ["bulk", "particle", "flux"]

    def compute_residual(
        self, t: float, y: np.ndarray, y_dot: np.ndarray, residual: np.ndarray
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
