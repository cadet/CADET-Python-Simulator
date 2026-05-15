from typing import NoReturn, Optional

import numpy as np
import numpy.typing as npt
import xarray as xr
from addict import Dict
from CADETProcess.dataStructure import Structure
from scikits.odes.dae import dae

from CADETPythonSimulator.exception import CADETPythonSimError
from CADETPythonSimulator.system import SystemBase
from CADETPythonSimulator.unit_operation import UnitOperationBase


class Solver(Structure):
    """Solver Class to solve a System."""

    def __init__(self, system: SystemBase, sections: list[dict]):
        """Construct the Solver Class."""
        self.initialize_solver()
        self._system: Optional[SystemBase] = system
        self._setup_sections(sections)

    def _setup_sections(self, sections):
        """
        Set up sections.

        Converts dict sections into addict.Dicts. Checks if start and end times
        are continusly
        """
        for i, section in enumerate(sections):
            if type(section) is dict:
                sections[i] = Dict(section)

        previous_end = sections[0].start
        for section in sections:
            if section.end <= section.start:
                raise ValueError("Section end must be larger than section start.")
            if section.start != previous_end:
                raise ValueError("Sections times must be without gaps.")
            previous_end = section.end
        self.sections = sections

    def initialize_solver(self, solver: str = "ida") -> NoReturn:
        """
        Initialize solver.

        Parameters
        ----------
        solver : str, optional
            Solver to use for integration. The default is `ida`.

        """
        if solver not in ["ida"]:
            raise ValueError(f"{solver} is not a supported solver.")

        self.solver = dae(solver, self._compute_residual)

    @property
    def system(self) -> SystemBase:
        """Return System of Object."""
        return self._system

    @system.setter
    def system(self, system: SystemBase) -> NoReturn:
        """Setter for System."""
        self._system = system

    def _compute_residual(
        self, t: float, y: np.ndarray, y_dot: np.ndarray, r: np.ndarray
    ) -> NoReturn:
        """
        Compute residual of the system.

        Parameters
        ----------
        t : float
            Time to evaluate
        y : np.ndarray
            State to evaluate
        y_dot : np.ndarray
            State derivative to evaluate
        r : np.ndarray
            Array to save the calculated residual

        """
        self._system.compute_residual(t, y, y_dot)
        r[...] = self._system.r

    def initialize_solution_recorder(self) -> NoReturn:
        """Initialize per-entry accumulation lists for all unit operation states."""
        self._time_list: list[float] = []
        self._y_lists: dict = {}
        self._yd_lists: dict = {}
        self._y_aux_lists: dict = {}

        for unit in self._system.unit_operations.values():
            self._y_lists[unit.name] = {
                s: {e: [] for e in state.entries} for s, state in unit.states.items()
            }
            self._yd_lists[unit.name] = {
                s: {e: [] for e in state.entries}
                for s, state in unit.state_derivatives.items()
            }
            self._y_aux_lists[unit.name] = {
                s: {e: [] for e in state.entries}
                for s, state in unit.aux_states.items()
            }

    def write_solution(
        self, times: np.ndarray, y_history: np.ndarray, y_dot_history: np.ndarray
    ) -> NoReturn:
        """Snapshot all state entries at each time step and append to accumulation lists.

        Sets system.y / y_dot for each step so state views reflect the correct values,
        then restores the system to its state before this call.
        """
        saved_y = self._system.y.copy()
        saved_yd = self._system.y_dot.copy()

        for i, t in enumerate(times):
            self._system.y = y_history[i]
            self._system.y_dot = y_dot_history[i]
            self._time_list.append(float(t))

            for unit in self._system.unit_operations.values():
                unit.compute_auxiliary(t)

                for state_name, state in unit.states.items():
                    for entry in state.entries:
                        self._y_lists[unit.name][state_name][entry].append(
                            state.field[entry].values.copy()
                        )
                for state_name, state in unit.state_derivatives.items():
                    for entry in state.entries:
                        self._yd_lists[unit.name][state_name][entry].append(
                            state.field[entry].values.copy()
                        )
                for state_name, state in unit.aux_states.items():
                    for entry in state.entries:
                        self._y_aux_lists[unit.name][state_name][entry].append(
                            state.field[entry].values.copy()
                        )

        self._system.y = saved_y
        self._system.y_dot = saved_yd

    @property
    def time_solutions(self) -> np.ndarray:
        """Time points at which solution snapshots were recorded."""
        return np.array(self._time_list)

    @property
    def unit_solutions(self) -> dict[str, dict[str, dict[str, xr.DataArray]]]:
        """Recorded state values: unit → state → entry → DataArray(time, ...).

        Primary and auxiliary states appear at the same level.
        """
        return self._build_datasets(self._y_lists, self._y_aux_lists)

    @property
    def unit_derivatives(self) -> dict[str, dict[str, dict[str, xr.DataArray]]]:
        """Recorded state derivatives: unit → state → entry → DataArray(time, ...)."""
        return self._build_datasets(self._yd_lists)

    def _build_datasets(
        self,
        lists: dict,
        aux_lists: dict | None = None,
    ) -> dict[str, dict[str, dict[str, xr.DataArray]]]:
        times = self.time_solutions
        result = {}
        for unit in self._system.unit_operations.values():
            result[unit.name] = {}
            all_states = list(unit.states.items())
            if aux_lists is not None:
                all_states += list(unit.aux_states.items())
            for state_name, state in all_states:
                result[unit.name][state_name] = {}
                src = lists if state_name in lists[unit.name] else aux_lists
                sample = state.field
                for entry in state.entries:
                    da_sample = sample[entry]
                    stacked = np.stack(src[unit.name][state_name][entry], axis=0)
                    coords = {"time": times, **dict(da_sample.coords)}
                    dims = ["time"] + list(da_sample.dims)
                    result[unit.name][state_name][entry] = xr.DataArray(
                        stacked, coords=coords, dims=dims, name=entry
                    )
        return result

    def solve(self) -> NoReturn:
        """Simulate the system."""
        self.initialize_solution_recorder()
        self._system.wire_states()

        for section in self.sections[:-1]:
            self.solve_section(section)
        self.solve_section(self.sections[-1], True)

    def initialize_system(self):
        """Initialize System."""
        self.system.initialize_initial_values()

    def solve_section(
        self,
        section: Dict,
        last_section=False,
    ) -> NoReturn:
        """
        Solve a time section of the differential-algebraic equation system.

        This method uses the solver initialized by the system to compute the states and
        their derivatives over a specified section.

        Parameters
        ----------
        section : Dict
            The time points at which the solution is sought.
        last_section : Bool
            Parameter to check if calculating the last section for consistent issues
        #TODO: Consider creating a section class instead of using Addict

        """
        self._update_unit_operation_parameters(
            section.start,
            section.end,
            section.section_states,
        )
        self._system.set_connections(section.connections)
        self._system.initialize_initial_values(section.start)

        section_solution_times = self.get_section_solution_times(section)

        y_initial = self._system.y
        y_initial_dot = self._system.y_dot
        output = self.solver.solve(section_solution_times, y_initial, y_initial_dot)

        y_history = output.values.y
        y_dot_history = output.values.ydot
        times = output.values.t

        self._system.y = y_history[-1]
        self._system.y_dot = y_dot_history[-1]

        if not last_section:
            self.write_solution(times[:-1], y_history[:-1], y_dot_history[:-1])
        else:
            self.write_solution(times, y_history, y_dot_history)

    def get_section_solution_times(self, section: Dict) -> np.ndarray:
        """
        Calculate Section Times.

        Every Section describes the solution times with a start and end time.w
        The solution times are simply calculated by using np.arange with time_resolution
        as step size. If time_resolution is not given, a resolution of 1s is used.

        Parameters
        ----------
        section : Dict
            Section dict that describes the Parameters

        """
        # TODO: How to get section_solution_times from section.start, section.end, if
        # user_solution times are provided?
        time_resolution = section.get("time_resolution", 1.0)
        start = section.start
        end = section.end + time_resolution
        return np.arange(start, end, time_resolution)

    def _update_unit_operation_parameters(
        self,
        start: float,
        end: float,
        unit_operation_parameters: dict[
            UnitOperationBase | str, dict[str, npt.ArrayLike]
        ],
    ) -> np.ndarray:
        """
        Update time dependent unit operation parameters.

        Parameters
        ----------
        start: float
            Start time of the section.
        end: float
            End time of the section.
        unit_operation_parameters : dict[UnitOperation | str, dict]
            Unit operation parameters for the next section.

        """
        for unit, parameters in unit_operation_parameters.items():
            if isinstance(unit, str):
                try:
                    unit = self._system.unit_operations[unit]
                except KeyError:
                    raise CADETPythonSimError(f"Unit {unit} is not Part of the System.")
            else:
                if unit not in self._system.unit_operations.values():
                    raise CADETPythonSimError(f"Unit {unit} is not Part of the System.")

            unit.update_parameters(start, end, parameters)
