from collections import defaultdict
from dataclasses import dataclass
from typing import NoReturn, Optional

from addict import Dict
import numpy as np

from CADETPythonSimulator.state import State
from CADETPythonSimulator.coupling_interface import (
    CouplingInterface,
    WeightedAverageCoupling,
)

from CADETProcess.dataStructure import Structure
from CADETPythonSimulator.exception import NotInitializedError, CADETPythonSimError
from CADETPythonSimulator.unit_operation import UnitOperationBase
from CADETPythonSimulator.componentsystem import CPSComponentSystem


@dataclass
class Connection:
    """A directed flow connection between two unit operation ports."""

    source: str
    source_port: str
    dest: str
    dest_port: str
    flow_rate: float
    source_index: int | None = None
    dest_index: int | None = None


class SystemBase(Structure):
    """Base Class Structure for a System."""

    def __init__(self, unit_operations: list[UnitOperationBase]):
        """Construct the SystemBase Class."""
        self._states: Optional[dict[str, State]] = None
        self._state_derivatives: Optional[dict[str, State]] = None
        self._residuals: Optional[dict[str, State]] = None
        self._component_system: Optional[CPSComponentSystem] = None

        self._t_zero: Optional[float] = None

        self._y_buf: Optional[np.ndarray] = None
        self._yd_buf: Optional[np.ndarray] = None
        self._r_buf: Optional[np.ndarray] = None

        self._connections: list[Connection] = []

        if not self._coupling_state_structure:
            self._coupling_state_structure: Optional[dict[str, CouplingInterface]] = (
                None
            )
        self._setup_unit_operations(unit_operations)

    @property
    def unit_operations(self) -> dict[str, UnitOperationBase]:
        """dict: Unit operations, indexed by name."""
        return self._unit_operations

    @property
    def n_dof(self) -> int:
        """int: Total number of degrees of freedom."""
        return sum(
            [unit_operation.n_dof for unit_operation in self.unit_operations.values()]
        )

    @property
    def n_comp(self) -> int:
        """int: Number of components."""
        return self._component_system.n_comp

    def initialize_state(self) -> NoReturn:
        """Initialize the system state and residual."""
        self._states = {}
        self._state_derivatives = {}
        self._residuals = {}
        for unit_name, unit_operation in self.unit_operations.items():
            unit_operation.initialize_state()
            self._states[unit_name] = unit_operation.states
            self._state_derivatives[unit_name] = unit_operation.state_derivatives
            self._residuals[unit_name] = unit_operation.residuals

    def wire_states(self) -> None:
        """Wire all state buffers to three flat system-owned arrays.

        Called once by the Solver before the first section. After this,
        system.y / y_dot / r are single bulk copies instead of N cascaded
        per-state copies, and system.r requires no concatenation.
        """
        n = self.n_dof
        # Snapshot current values before replacing _s pointers
        y_current = self.y
        yd_current = self.y_dot

        self._y_buf = np.empty(n)
        self._yd_buf = np.empty(n)
        self._r_buf = np.empty(n)

        self._y_buf[...] = y_current
        self._yd_buf[...] = yd_current

        offset = 0
        for unit in self.unit_operations.values():
            for name in unit.states:
                n_state = unit.states[name].n_dof
                unit.states[name].wire_to_buffer(self._y_buf, offset)
                unit.state_derivatives[name].wire_to_buffer(self._yd_buf, offset)
                unit.residuals[name].wire_to_buffer(self._r_buf, offset)
                offset += n_state

    def _setup_unit_operations(self, unit_operations: list[UnitOperationBase]):
        """
        Set up unit operations.

        Checks whether every Unit has the same Component system and adds them to a
        dictionary.

        Parameters
        ----------
        unit_operations : list[UnitOperationBase]
            list of UnitOperation that are part of the System

        """
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
        if self._y_buf is not None:
            return self._y_buf
        return np.concatenate(
            [unit_operation.y for unit_operation in self.unit_operations.values()]
        )

    @y.setter
    def y(self, y: np.ndarray) -> NoReturn:
        """Setter for state in Form of y array."""
        if self._y_buf is not None:
            self._y_buf[...] = y
            return
        start_index = 0
        for unit_operation in self.unit_operations.values():
            end_index = start_index + unit_operation.n_dof
            unit_operation.y = y[start_index:end_index]
            start_index = end_index

    @property
    def y_init(self) -> np.ndarray:
        """np.ndarray: State array flattened into one dimension."""
        return np.concatenate(
            [unit_operation.y_init for unit_operation in self.unit_operations.values()]
        )

    @y_init.setter
    def y_init(self, y_init: np.ndarray) -> NoReturn:
        """Setter for state in Form of y_init array."""
        start_index = 0
        for unit_operation in self.unit_operations.values():
            end_index = start_index + unit_operation.n_dof
            unit_operation.y_init = y_init[start_index:end_index]
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
        if self._yd_buf is not None:
            return self._yd_buf
        return np.concatenate(
            [unit_operation.y_dot for unit_operation in self.unit_operations.values()]
        )

    @y_dot.setter
    def y_dot(self, y_dot: np.ndarray) -> NoReturn:
        """Setter for y_dot."""
        if self._yd_buf is not None:
            self._yd_buf[...] = y_dot
            return
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
        if self._r_buf is not None:
            return self._r_buf
        return np.concatenate(
            [unit_operation.r for unit_operation in self.unit_operations.values()]
        )

    @r.setter
    def r(self, r: np.ndarray) -> NoReturn:
        """Setter for r."""
        if self._r_buf is not None:
            self._r_buf[...] = r
            return
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
    def t_zero(self) -> float:
        """float: Initial Time of current Section."""
        return self._t_zero

    @t_zero.setter
    def t_zero(self, t0: float) -> NoReturn:
        """Setter for Initial Time of current Section."""
        self._t_zero = t0

    # ── connection API ────────────────────────────────────────────────────

    def set_connections(self, connections: list) -> None:
        """Parse and store connections for the current section."""
        self._connections = [self._parse_connection(c) for c in connections]

    def _resolve_endpoint(
        self, endpoint, direction: str
    ) -> tuple[str, str, int | None]:
        """Return (unit_name, port_name, port_index) from a connection endpoint.

        Accepts a plain unit name (str), a (unit, port) tuple, or a
        (unit, port, index) tuple for vector ports.
        """
        if isinstance(endpoint, str):
            unit_name = endpoint
            unit = self.unit_operations[unit_name]
            ports = unit._outlet_ports if direction == "outlet" else unit._inlet_ports
            if len(ports) == 1:
                return unit_name, next(iter(ports)), None
            if len(ports) == 0:
                raise ValueError(
                    f"Unit '{unit_name}' has no {direction} ports."
                )
            raise ValueError(
                f"Unit '{unit_name}' has multiple {direction} ports "
                f"{list(ports)}: specify explicitly as (unit_name, port_name)."
            )
        if isinstance(endpoint, (tuple, list)):
            if len(endpoint) == 2:
                return str(endpoint[0]), str(endpoint[1]), None
            if len(endpoint) == 3:
                return str(endpoint[0]), str(endpoint[1]), int(endpoint[2])
            raise ValueError(f"Invalid endpoint spec: {endpoint!r}")
        raise TypeError(f"Invalid endpoint type: {type(endpoint)}")

    def _parse_connection(self, conn) -> Connection:
        """Parse one connection spec into a Connection object."""
        if isinstance(conn, Connection):
            return conn
        if isinstance(conn, dict):
            src_ep = conn["source"]
            dst_ep = conn["dest"]
            flow_rate = float(conn["flow_rate"])
        else:
            src_ep, dst_ep, flow_rate = conn[0], conn[1], float(conn[2])
        src_name, src_port, src_idx = self._resolve_endpoint(src_ep, "outlet")
        dst_name, dst_port, dst_idx = self._resolve_endpoint(dst_ep, "inlet")
        return Connection(
            source=src_name,
            source_port=src_port,
            dest=dst_name,
            dest_port=dst_port,
            flow_rate=flow_rate,
            source_index=src_idx,
            dest_index=dst_idx,
        )

    def update_ports(self) -> None:
        """Set flow rates and inlet states from current connections.

        Replaces the old couple_unit_operations() + set_rates() pair.
        """
        # Reset all flow rates
        for unit in self.unit_operations.values():
            if unit._Q_in is not None:
                unit._Q_in[:] = 0.0
            if unit._Q_out is not None:
                unit._Q_out[:] = 0.0

        if not self._connections:
            return

        # Accumulate Q_out per source port
        for conn in self._connections:
            src_unit = self.unit_operations[conn.source]
            src_idx = list(src_unit._outlet_ports).index(conn.source_port)
            src_unit._Q_out[src_idx] += conn.flow_rate

        # Group by destination port and do weighted-average state coupling
        by_dest: dict[tuple, list] = defaultdict(list)
        for conn in self._connections:
            by_dest[(conn.dest, conn.dest_port)].append(conn)

        for (dest_name, dest_port), conns in by_dest.items():
            dest_unit = self.unit_operations[dest_name]
            total_Q = sum(c.flow_rate for c in conns)

            dst_idx = list(dest_unit._inlet_ports).index(dest_port)
            dest_unit._Q_in[dst_idx] = total_Q

            if total_Q > 0:
                unit_Q_list = [
                    (
                        self.unit_operations[c.source].get_port_state(
                            c.source_port, c.source_index or 0
                        ),
                        c.flow_rate,
                    )
                    for c in conns
                ]
                coupled = self.coupled_state_func(unit_Q_list)
                dest_unit.set_port_state(
                    dest_port, coupled, conns[0].dest_index or 0
                )

    def compute_residual(self, t: float, y: np.ndarray, y_dot: np.ndarray) -> NoReturn:
        """
        Compute the residual for the differential-algebraic equations system.

        Parameters
        ----------
        t : float
            Current time point.
        y : np.ndarray
            initial state
        y_dot : np.ndarray
            initial state derivative

        """
        self.y = y
        self.y_dot = y_dot
        self.r = y
        self.update_ports()
        for unit_operation in self.unit_operations.values():
            unit_operation.compute_residual(t)

    def coupled_state_func(self, unit_Q_list: list[dict, float]) -> dict:
        """
        Create new state.

        Creates the new coupled state for a given Unit Operation by Iterating over the
        coupling state structure and their coupling Interface.

        Parameters
        ----------
        unit_Q_list : list[dict, float]
            list that contains every state and their corresponding rate that is connected
            to the to create state

        """
        ret = {}
        for state, calc_method in self.coupling_state_structure.items():
            ret[state] = calc_method.get_coupled_state(unit_Q_list, state)
        return ret

    def initialize_initial_values(self, t0: float):
        """
        Calculate initial values and set them up.

        First iterates over every Unit Operation to set their initial values up and
        then tries to calculate a consistent one.

        """
        self.t_zero = t0
        self.update_ports()
        for unit_operation in self.unit_operations.values():
            unit_operation.initialize_initial_values(t0)
            self.update_ports()


class FlowSystem(SystemBase):
    """
    SystemBase Class.

    Class that implements the SystemBase with standard Flowrates
    inbetween Unit Operations.
    """

    def __init__(self, unit_operations: list[UnitOperationBase]):
        """Construct FlowSystem Object."""
        self.coupling_state_structure = {"c": WeightedAverageCoupling()}
        super().__init__(unit_operations)
