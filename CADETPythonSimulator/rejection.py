from abc import abstractmethod

from CADETProcess.dataStructure import Structure, UnsignedFloat


class RejectionBase(Structure):
    """Base class for rejection curves."""

    @abstractmethod
    def get_rejection(self, molar_mass: float) -> float:
        """
        Return rejection for a species.

        Parameters
        ----------
        molar_mass : float
            Molar mass [kg/mol]. Numerically equal to kDa.

        Returns
        -------
        float
            Rejection in [0, 1].
        """
        pass


class StepCutOff(RejectionBase):
    """
    Step rejection model: retain all species at or above a molar mass cutoff.

    Attributes
    ----------
    cutoff_weight : float
        Cutoff molar mass [kg/mol = kDa]. Species with molar_mass >= cutoff_weight
        are fully retained; species below pass through.
    """

    cutoff_weight = UnsignedFloat()

    def get_rejection(self, molar_mass: float) -> float:
        """Return 0 (pass) if molar_mass < cutoff_weight, else 1 (retain)."""
        return 0 if molar_mass < self.cutoff_weight else 1
