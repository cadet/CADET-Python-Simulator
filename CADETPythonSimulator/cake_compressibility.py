from abc import abstractmethod

from CADETProcess.dataStructure import Structure
from CADETProcess.dataStructure import UnsignedFloat


class CakeCompressibilityBase(Structure):
    """Base class to describe filter cake compressibility behaviour."""

    @abstractmethod
    def specific_resistance(self, delta_p: float) -> float:
        """
        Calculate specific cake resistance as a function of pressure difference.

        Parameters
        ----------
        delta_p : float
            Pressure difference.

        Returns
        -------
        float
            Specific pressure difference.

        """
        return


class NoCakeCompressibility(CakeCompressibilityBase):
    """
    Class to describe filter cake without compressibility behaviour.

    Attributes
    ----------
    cake_resistance: float
        Constant value of cake resistance factor.

    """

    cake_resistance = UnsignedFloat()

    def specific_resistance(self, delta_p: float) -> float:
        """
        Calculate specific cake resistance as a function of pressure difference.

        Parameters
        ----------
        delta_p : float
            Pressure difference.

        Returns
        -------
        float
            Specific pressure difference.

        """
        return self.cake_resistance


class LinearCakeCompressibility(CakeCompressibilityBase):
    """
    Class for filter cake with linearly increasing cake  behaviour.

    Attributes
    ----------
    cake_resistance_base: float
        Base value of cake resistance factor.
    cake_resistance_linear: float
        Slope of cake resistance factor.

    """

    cake_resistance_base = UnsignedFloat()
    cake_resistance_linear = UnsignedFloat()

    def specific_resistance(self, delta_p: float) -> float:
        """
        Calculate specific cake resistance as a function of pressure difference.

        Parameters
        ----------
        delta_p : float
            Pressure difference.

        Returns
        -------
        float
            Specific pressure difference.

        """
        return self.cake_resistance_base + self.cake_resistance_linear * delta_p
