from abc import abstractmethod

import numpy as np

from CADETProcess.dataStructure import Structure


class ViscosityBase(Structure):
    """Base class for mixed viscosity calculations."""

    @abstractmethod
    def get_mixture_viscosity(
            self, viscosities: np.ndarray, fractions: np.ndarray
            ) -> float:
        """Calculate mixed viscosity with given viscosities and volume fractions.

        Parameters
        ----------
        viscosities : np.ndarray
            List of viscosities of the components.
        fractions : np.ndarray
            List of volume fractions of the components.

        Returns
        -------
        float
            Calculated mixed viscosity of the mixture.
        """
        pass

    def _validate_viscosities_input(
            self,
            viscosities: np.ndarray,
            fractions: np.ndarray
            ) -> None:
        if not viscosities or not fractions or len(viscosities) != len(fractions):
            raise ValueError(
                "Viscosities and fractions lists must be of the same length."
            )
        if not np.isclose(np.sum(fractions), 1):
            raise ValueError(
                "Sum of volume fractions must be 1."
            )


class AverageViscosity(ViscosityBase):
    """Calculate mixed viscosity using the average mean."""

    def get_mixture_viscosity(
            self, viscosities: np.ndarray, fractions: np.ndarray
            ) -> float:
        """Calculate mixed viscosity using the arithmetic mean.

        Parameters
        ----------
        viscosities : np.ndarray
            List of viscosities of the components.
        fractions : np.ndarray
            List of volume fractions of the components.

        Returns
        -------
        float
            Weighted average mixed viscosity of the mixture.
        """
        self._validate_viscosities_input(viscosities, fractions)

        mixed_viscosity = sum(v * f for v, f in zip(viscosities, fractions))
        return mixed_viscosity


class LogarithmicMixingViscosity(ViscosityBase):
    """Calculate mixed viscosity using the logarithmic mixing rule."""

    def get_mixture_viscosity(
            self, viscosities: np.ndarray, fractions: np.ndarray
            ) -> float:
        """Calculate mixed viscosity using the logarithmic mixing rule.

        Parameters
        ----------
        viscosities : np.ndarray
            List of viscosities of the components.
        fractions : np.ndarray
            List of volume fractions of the components.

        Returns
        -------
        float
            Logarithmic mixed viscosity of the mixture.
        """
        self._validate_viscosities_input(viscosities, fractions)

        mixed_viscosity = np.exp(
            np.sum(f * np.log(v) for v, f in zip(viscosities, fractions))
        )
        return mixed_viscosity
