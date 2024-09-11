import abc
import numpy as np
from CADETPythonSimulator.unit_operation import UnitOperationBase


class CouplingInterface(abc.ABC):
    """Abstract interface for coupling states."""

    @abc.abstractmethod
    def get_coupled_state(
        self,
        origin_list: list[(dict, float)],
        state: str
        ) -> np.ndarray:
        """Calculate new state for destination_unit."""


class WeightedAverageCoupling(CouplingInterface):
    """Implements the Coupling Interface for average Coupling."""

    def get_coupled_state(self,
                          origin_list: list[(dict, float)],
                          state: str
                          ) -> np.ndarray:
        """Calculate new state for destination_unit with average Coupling."""
        ret = np.zeros(origin_list[0][0][state].shape)
        rate_tot = 0
        for list_state, rate in origin_list:
            ret += list_state[state] * rate
            rate_tot += rate

        ret /= rate_tot

        return ret
