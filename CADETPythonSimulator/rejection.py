from abc import abstractmethod

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import UnsignedFloat


class RejectionBase(Structure):
    """Base class for rejection curves."""

    @abstractmethod
    def get_rejection(self, mw: float) -> float:
        """Get rejection for a species with specific molecular weight.

        Parameters
        ----------
        mw : float
            Molecular weight.

        Returns
        -------
        float
            Rejection for given specific molecular weight.

        """
        pass


class StepCutOff(RejectionBase):
    """
    Rejection model for step cutoff size.

    Attributes
    ----------
    cutoff_weight : float
        Cutoff size. All molecules smaller than the size will pass through the filter.
        All molecules larger or equal than the cutoff will be retained.
    """

    cutoff_weight = UnsignedFloat()

    def get_rejection(self, mw: float) -> float:
        """Get rejection for a species with specific molecular weight.

        Parameters
        ----------
        mw : float
            Molecular weight.

        Returns
        -------
        float
            Rejection for given specific molecular weight.

        """
        return 0 if mw < self.cutoff_weight else 1
