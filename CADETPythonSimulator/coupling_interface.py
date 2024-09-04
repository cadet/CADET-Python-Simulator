import abc
import numpy as np
from CADETPythonSimulator.unit_operation import UnitOperationBase


class CouplingInterface(abc.ABC):
    @abc.abstractmethod
    def get_coupled_state(self,
                          origin_list: list[(dict, float)],
                          coupling_state_structure: dict
                          ) -> dict:
        """Calculates new state for destination_unit"""



class AverageCoupling(CouplingInterface):

    def get_coupled_state(self,
                          origin_list: list[(dict, float)],
                          coupling_state_structure: dict
                          ) -> dict:
        ret = {}
        for state, dim in coupling_state_structure.items():
            ret[state] = np.zeros(dim)

        value_tot = 0
        for unit, value in origin_list:
            for state in coupling_state_structure.keys():
                ret[state] += unit[state]*value
            value_tot += value

        for value in ret.values():
            value/=value_tot

        return ret