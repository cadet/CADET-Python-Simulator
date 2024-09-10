from typing import NoReturn, Optional

from addict import Dict
import numpy as np
import numpy.typing as npt
from scikits.odes.dae import dae

from CADETProcess.dataStructure import Structure
from CADETPythonSimulator.exception import NotInitializedError
from CADETPythonSimulator.unit_operation import UnitOperationBase
from CADETPythonSimulator.system import SystemBase


class Solver(Structure):
    """Solver Class to solve a System."""

    def __init__(self, system: SystemBase, sections: list[dict]):
        """Construct the Solver Class."""
        self.initialize_solver()
        self._system: Optional[SystemBase] = system
        self._setup_sections(sections)


    def _setup_sections(self, sections):
        # TODO: Check section continuity.

        self.sections = sections


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

        self.solver = dae(solver, self._compute_residual)

    @property
    def system(self) -> SystemBase:
        """Return System of Object."""
        return self._system

    @system.setter
    def system(self, system:SystemBase) -> NoReturn:
        """Setter for System."""
        self._system = system

    def _compute_residual(
            self,
            t: float,
            y: np.ndarray,
            y_dot: np.ndarray,
            r: np.ndarray
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
        self._system.y = y
        self._system.y_dot = y_dot
        self._system.compute_residual(t)
        r[...] = self._system.r

    def initialize_solution_recorder(self) -> NoReturn:
        """
        Initialize the solution recorder for all unit_operations.

        Iterates over each unit in the system and initializes an empty numpy array for
        each state variable within the unit. The structure and size of the array for
        each state variable are determined by the unit's state structure.
        """
        self.unit_solutions: dict[str, dict] = {}
        self.time_solutions: np.ndarray = np.empty((0,))

        for unit in self._system.unit_operations.values():
            self.unit_solutions[unit.name]: dict[str, dict] = {}
            for state_name, state in unit.states.items():
                self.unit_solutions[unit.name][state_name]: dict[str, dict] = {}
                for entry, dim in state.entries.items():
                    self.unit_solutions[unit.name][state_name][entry]:\
                        dict[str, np.ndarray] = {}
                    self.unit_solutions[unit.name][state_name][entry]['values'] =\
                        np.empty((0, dim))
                    self.unit_solutions[unit.name][state_name][entry]['derivatives'] =\
                        np.empty((0, dim))

    def write_solution(
            self,
            times: np.ndarray,
            y_history: np.ndarray,
            y_dot_history: np.ndarray
        ) -> NoReturn:
        """
        Update the solution recorder for each unit with the current state.

        Iterates over each unit, to extract the relevant portions of `y` and `y_dot`.
        The current state of each unit is determined by splitting `y` according to each
        unit's requirements.
        """
        it = 0
        for state in self.unit_solutions.values():
            for state_dict in state.values():
                for sol_tuple in state_dict.values():
                    itp = it + sol_tuple["values"].shape[1]

                    y = y_history[:,it:itp]
                    ydot = y_dot_history[:,it:itp]

                    sol_tuple["values"] = np.concatenate((sol_tuple["values"], y))
                    sol_tuple["derivatives"] = np.concatenate((
                        sol_tuple["derivatives"], ydot
                    ))
                    it = itp

        self.time_solutions = np.concatenate((self.time_solutions, times))

    def solve(self) -> NoReturn:
        """Simulate the system."""
        self.initialize_system()
        self.initialize_solution_recorder()

        previous_end = self.sections[0].start
        for section in self.sections:
            if section.start <= section.end:
                raise ValueError("Section end must be larger than section start.")
            if section.start != previous_end:
                raise ValueError("Sections times must be without gaps.")

            self.solve_section(section)
            self.write_solution()

    def initialize_system(self):
        """Initialize System."""
        self.system.initialize()


    def solve_section(
            self,
            section: Dict,
            ) -> NoReturn:
        """
        Solve a time section of the differential-algebraic equation system.

        This method uses the solver initialized by the system to compute the states and
        their derivatives over a specified section.

        Parameters
        ----------
        section : Dict
            The time points at which the solution is sought.
        #TODO: Consider creating a section class instead of using Addict

        """
        self._update_unit_operation_parameters(
            section.start,
            section.end,
            section.section_states,
        )
        self._system.update_system_connectivity(section.connections)

        section_solution_times = self.get_section_solution_times(section)

        y_initial = self._system.y
        y_initial_dot = self._system.y_dot
        output = self.solver.solve(
            section_solution_times, y_initial, y_initial_dot
        )

        y_history = output.values.y
        y_dot_history = output.values.ydot
        times = output.values.t

        self._system.y = y_history[-1]
        self._system.y_dot = y_dot_history[-1]

        self.write_solution(times, y_history, y_dot_history)

    def get_section_solution_times(self, section: Dict) -> np.ndarray:
        # TODO: How to get section_solution_times from section.start, section.end, if user_solution times are provided?
        raise NotImplementedError()

    def _update_unit_operation_parameters(
            self,
            start: float,
            end: float,
            unit_operation_parameters: dict[
                UnitOperationBase | str,
                dict[str, npt.ArrayLike]
                ]
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
            # TODO: Check if unit is part of SystemSolver
            if isinstance(unit, str):
                unit = self.unit_operations_dict[unit]

            unit.update_section_dependent_parameters(start, end, parameters)

    # @property
    # def port_mapping(self) -> dict[int, str]:
    #     """dict: Mapping of port indices to corresponding state entries."""
    #     # TODO: Let this be handled by the SystemSolver?
    #     port_mapping = defaultdict(dict)

    #     counter = 0
    #     for mapped_state, n_ports in self.inlet_ports.items():
    #         for port in range(n_ports):
    #             port_mapping['inlet'][counter] = {}
    #             port_mapping['inlet'][counter]['mapped_state'] = mapped_state
    #             port_mapping['inlet'][counter]['port_index'] = port
    #             counter += 1

    #     counter = 0
    #     for mapped_state, n_ports in self.outlet_ports.items():
    #         for port in range(n_ports):
    #             port_mapping['outlet'][counter] = {}
    #             port_mapping['outlet'][counter]['mapped_state'] = mapped_state
    #             port_mapping['outlet'][counter]['port_index'] = port
    #             counter += 1

    #     return dict(port_mapping)
