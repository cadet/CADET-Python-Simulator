from typing import NoReturn, Optional

import abc
from addict import Dict
import numpy as np
import numpy.typing as npt
from scikits.odes.dae import dae

from CADETPythonSimulator.state import State, state_factory
from CADETPythonSimulator.coupling_interface import CouplingInterface, AverageCoupling

from CADETProcess.dataStructure import Structure
from CADETPythonSimulator.exception import NotInitializedError, CADETPythonSimError
from CADETPythonSimulator.unit_operation import UnitOperationBase
from CADETPythonSimulator.componentsystem import CPSComponentSystem

class SystemBase(Structure):
    """Base Class Structure for a System."""

    def __init__(self, unit_operations: list[UnitOperationBase]):
        """Construct the SystemBase Class."""
        self._states: Optional[dict[str, State]] = None
        self._state_derivatives: Optional[dict[str, State]] = None
        self._residuals: Optional[dict[str, State]] = None
        self._component_system: Optional[CPSComponentSystem] = None
        self._connectivity: Optional[np.ndarray] = None

        if not self._coupling_state_structure:
            self._coupling_state_structure: Optional[dict[str, CouplingInterface]]\
                = None
        self._setup_unit_operations(unit_operations)


    @property
    def unit_operations(self) -> dict[str, UnitOperationBase]:
        """dict: Unit operations, indexed by name."""
        return self._unit_operations

    @property
    def n_dof(self) -> int:
        """int: Total number of degrees of freedom."""
        return sum([
            unit_operation.n_dof for unit_operation in self.unit_operations.values()
        ])

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self._component_system.n_comp

    def initialize(self) -> NoReturn:
        """Initialize the system state and residual."""
        self._states = {}
        self._state_derivatives = {}
        self._residuals = {}
        for unit_name, unit_operation in self.unit_operations.items():
            unit_operation.initialize()
            self._states[unit_name] = unit_operation.states
            self._state_derivatives[unit_name] = unit_operation.state_derivatives
            self._residuals[unit_name] = unit_operation.residuals

        self._setup_connectivity()

    def _setup_connectivity(self):
        """Set up helper Parameters for connectivity."""
        # dict with [origin_index]{unit, port}
        origin_index_unit_operations = Dict()
        # Nested dict with [unit_operations][ports]: origin_index in connectivity matrix
        origin_unit_ports = Dict()
        origin_counter = 0
        for i_unit, unit in enumerate(self.unit_operations.items()):
            for port in range(unit[1].n_outlet_ports):
                origin_unit_ports[i_unit][port] = origin_counter
                origin_index_unit_operations[origin_counter] = {
                    'unit': i_unit,
                    'port': port,
                    'name': unit[0],
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
        for i_unit, unit in enumerate(self.unit_operations.items()):
            for port in range(unit[1].n_inlet_ports):
                destination_unit_ports[i_unit][port] = destination_counter
                destination_index_unit_operations[destination_counter] = {
                    'unit': i_unit,
                    'port': port,
                    'name': unit[0],
                }
                destination_counter += 1
        self.destination_unit_ports = destination_unit_ports
        self.destination_index_unit_operations = destination_index_unit_operations
        self.n_destination_ports = destination_counter

    def _setup_unit_operations(self, unit_operations : list[UnitOperationBase]):
        #TODO: check if all unit_operation satisfy the system
        self._component_system = unit_operations[0].component_system

        for unit in unit_operations:
            if unit.component_system is not self._component_system:
                raise CADETPythonSimError(
                    f"""Unit Operation {unit} has a different
                    Component System than the first unit operation
                    {unit_operations[0]}."""
                )
        self._unit_operations = {unit.name: unit for unit in unit_operations}

    @property
    def states(self) -> dict[str, dict[str, State]]:
        """dict: State array block of the system, indexed by unit operation name."""
        if self._states is None:
            raise NotInitializedError("System state is not yet initialized.")

        return self._states

    @property
    def y(self) -> np.ndarray:
        """np.ndarray: State array flattened into one dimension."""
        return np.concatenate([
            unit_operation.y for unit_operation in self.unit_operations.values()
        ])

    @y.setter
    def y(self, y: np.ndarray) -> NoReturn:
        start_index = 0
        for unit_operation in self.unit_operations.values():
            end_index = start_index + unit_operation.n_dof
            unit_operation.y = y[start_index:end_index]
            start_index = end_index

    @property
    def state_derivatives(self) -> dict[str, dict[str, State]]:
        """dict: State derivative array block of the system, indexed by name."""
        if self._state_derivatives is None:
            raise NotInitializedError("System state is not yet initialized.")

        return self._state_derivatives

    @property
    def y_dot(self) -> np.ndarray:
        """np.ndarray: State derivative array flattened into one dimension."""
        return np.concatenate([
            unit_operation.y_dot for unit_operation in self.unit_operations.values()
        ])

    @y_dot.setter
    def y_dot(self, y_dot: np.ndarray) -> NoReturn:
        start_index = 0
        for unit_operation in self.unit_operations.values():
            end_index = start_index + unit_operation.n_dof
            unit_operation.y_dot = y_dot[start_index:end_index]
            start_index = end_index

    @property
    def residuals(self) -> dict[str, dict[str, State]]:
        """list: Residual array blocks of the system."""
        if self._residuals is None:
            raise NotInitializedError("System residual is not yet initialized.")

        return self._residuals

    @property
    def r(self) -> np.ndarray:
        """np.ndarray: Residual array flattened into one dimension."""
        return np.concatenate([
            unit_operation.r for unit_operation in self.unit_operations.values()
        ])

    @r.setter
    def r(self, r: np.ndarray) -> NoReturn:
        start_index = 0
        for unit_operation in self.unit_operations.values():
            end_index = start_index + unit_operation.n_dof
            unit_operation.r = r[start_index:end_index]
            start_index = end_index

    @property
    def coupling_state_structure(self) -> dict[str, CouplingInterface]:
        """dict: State structure that must be accessible in inlet / outlet ports."""
        return self._coupling_state_structure

    @coupling_state_structure.setter
    def coupling_state_structure(self, coupling_state_structure: dict) -> NoReturn:
        """Setter to set the coupling_state_structure."""
        self._coupling_state_structure = coupling_state_structure

    @property
    def connectivity(self) -> np.ndarray:
        """np.ndarray: Connectivity Matrix."""
        return self._connectivity

    def update_system_connectivity(self, connections: list) -> NoReturn:
        """
        Update the System connectivity.

        Parameters
        ----------
        connections : list
            list that contains the conncections between Unit Operations

        """
        self._compute_connectivity_matrix(connections)
        self.set_rates()
        self.couple_unit_operations()

    def set_rates(self) -> NoReturn:
        """
        Set Flowrates of Unit Operations.

        Parameters
        ----------
        flowrates: dict[str:np.ndarray]
            dictionary containing the rates

        """
        raise NotImplementedError("SystemBase is abstract.")

    def compute_residual(
            self,
            t: float
            ) -> NoReturn:
        """
        Compute the residual for the differential-algebraic equations system.

        Parameters
        ----------
        t : float
            Current time point.

        """
        for unit_operation in self.unit_operations.values():
            unit_operation.compute_residual(t)

    def couple_unit_operations(
            self,
            ) -> NoReturn:
        """Couple unit operations for set parameters."""
        for destination_port_index, Q_destinations in enumerate(self.connectivity):
            Q_destination_total = sum(Q_destinations)
            if Q_destination_total == 0:
                continue

            destination_info = \
                self.destination_index_unit_operations[destination_port_index]
            destination_unit = self.unit_operations[destination_info['name']]
            destination_port = destination_info['port']


            unit_Q_list = []
            for origin_port_index, Q_destination in enumerate(Q_destinations):
                if Q_destination == 0:
                    continue

                origin_info = self.origin_index_unit_operations[origin_port_index]
                origin_unit = self.unit_operations[origin_info['name']]
                origin_port = origin_info['port']

                unit_Q_list.append(
                    (origin_unit.get_outlet_state_flat(origin_port), Q_destination)
                    )

            s_new = self.coupled_state_func(unit_Q_list)

            destination_unit.set_inlet_state_flat(
                s_new, destination_port
            )

    def coupled_state_func(self, unit_Q_list: list[dict, float]) -> dict:
        """Create new state."""
        ret = {}
        for state, calc_method in self.coupling_state_structure.items():
            ret[state] = calc_method.get_coupled_state(unit_Q_list, state)
        return ret


    def _compute_connectivity_matrix(self, connections: list) -> np.ndarray:
        # TODO: This could be the setter for `connectivity`
        # Note, maybe we already adapt the interface s.t. we compute this matrix,
        # or we might even use a better maintainable data structure (e.g. a nested dict
        # representing the flow sheet as directed graph)
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
            destination_index =\
                self.destination_unit_ports[destination_unit][destination_port]

            rate = connection[4]

            connections_matrix[destination_index, origin_index] = rate

        self._connectivity = connections_matrix




class FlowSystem(SystemBase):
    """
    SystemBase Class.

    Class that implements the SystemBase with standard Flowrates
    inbetween Unit Operations.
    """

    def __init__(self, unit_operations: list[UnitOperationBase]):
        """Construct FlowSystem Object."""
        self.coupling_state_structure={
            'c': AverageCoupling(),
            'viscosity': AverageCoupling()
            }
        super().__init__(unit_operations)

    def set_rates(self) -> NoReturn:
        """
        Set Flowrates of Unit Operations.

        Parameters
        ----------
        flowrates: dict[str:np.ndarray]
            dictionary containing the rates

        """
        for dest_i, Q_n in enumerate(self.connectivity):
            unit_operation = self.destination_index_unit_operations[dest_i]['name']
            unit_port = self.destination_index_unit_operations[dest_i]['port']
            self.unit_operations[unit_operation].set_Q_in(unit_port, np.sum(Q_n))

        for origin_i, Q_n in enumerate(self.connectivity.T):
            unit_operation = self.origin_index_unit_operations[origin_i]['name']
            unit_port = self.origin_index_unit_operations[origin_i]['port']
            self.unit_operations[unit_operation].set_Q_out(unit_port, np.sum(Q_n))

