import pytest
from CADETPythonSimulator.componentsystem import Component, ComponentSystem
from CADETPythonSimulator.distribution_base import (
    ConstantConcentrationDistribution,
    ConstantVolumeDistribution,
)
from CADETPythonSimulator.exception import CADETPythonSimError

component_system = ComponentSystem(
    [
        Component("A", molar_mass=1.0, density=1.0, viscosity=1.0),
        Component("B", molar_mass=2.0, density=1.0, viscosity=1.0),
    ],
    name="test_comp",
)


@pytest.mark.parametrize(
    "distribution, component_system, concentration, expected",
    [
        (ConstantVolumeDistribution, component_system, [1], [1, 0]),
        (ConstantConcentrationDistribution, component_system, [1], [1, 0]),
    ],
)
class TestDistribution:
    """Test Class for Distribution."""

    def test_getter(self, distribution, component_system, concentration, expected):
        """Test return Function."""
        dist = distribution(component_system, concentration)
        assert dist.get_distribution(0, 0) == expected

    def test_error(self, distribution, component_system, concentration, expected):
        """Test Error Throwing."""
        with pytest.raises(ValueError):
            distribution(component_system, [1, 1])
        with pytest.raises(CADETPythonSimError):
            distribution(component_system, [10000])
