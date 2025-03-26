import numpy as np
import pytest

from CADETPythonSimulator.viscosity import (
    AverageViscosity,
    LogarithmicMixingViscosity,
    ViscosityBase,
)


class ViscosityDummy(ViscosityBase):
    """Dummy class to test Base functionalities."""

    def get_mixture_viscosity(self, viscosities, fractions):
        """Implement of Abstract Method."""
        pass

    def remove_nan_viscosities(self, viscosities, fractions):
        """Access to private Method."""
        return self._remove_nan_viscosities(viscosities, fractions)

    def validate_viscosities_input(self, viscosities, fractions):
        """Access to private Method."""
        return self._validate_viscosities_input(viscosities, fractions)


@pytest.mark.parametrize("Modul", [ViscosityDummy()])
class TestViscosityBase:
    """Class to test methods of Base Class."""

    def test_nan_viscosities(self, Modul):
        """Test removing nonviscosities Method."""
        viscosity_obj = Modul
        np.testing.assert_equal(
            viscosity_obj.remove_nan_viscosities(
                np.array([np.nan, 1, np.nan, 2]), np.array([0.25, 0.25, 0.25, 0.25])
            ),
            (np.array([1, 2]), np.array([0.5, 0.5])),
        )

    def test_validate_input(self, Modul):
        """Test validation Method."""
        viscosity_obj = Modul
        with pytest.raises(ValueError):
            viscosity_obj.validate_viscosities_input(np.array([]), np.array([1]))
        with pytest.raises(ValueError):
            viscosity_obj.validate_viscosities_input(np.array([0.5]), np.array([0.5]))

        viscosity_obj.validate_viscosities_input(
            np.array([0.5, 0.5]), np.array([0.5, 0.5])
        )


@pytest.mark.parametrize(
    "model, viscosities, fractions, expected",
    [
        (
            AverageViscosity(),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            1.5,
        ),
        (
            LogarithmicMixingViscosity(),
            np.array([1.0, 1.0, 2.0, 3.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.exp(0.25 * np.log(2) + 0.25 * np.log(3)),
        ),
    ],
)
class TestViscosityCalculation:
    """Testclass for viscosity calculation."""

    def test_viscosity_calculation(
        self,
        model: ViscosityBase,
        viscosities: np.ndarray,
        fractions: np.ndarray,
        expected: float,
    ):
        """Test Calculation."""
        np.testing.assert_allclose(
            model.get_mixture_viscosity(viscosities, fractions), expected
        )
