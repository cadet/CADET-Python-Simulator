import pytest

from CADETPythonSimulator.componentsystem import CPSComponentSystem
from CADETPythonSimulator.distribution_base import (
    ConstantVolumeDistribution,
    ConstantConcentrationDistribution,
)
from CADETPythonSimulator.exception import CADETPythonSimError

component_system = CPSComponentSystem(
    name="test_comp",
    components=2,
    densities=[1, 1],
    molecular_weights=[1, 2],
    viscosities=[1, 1],
    specific_cake_resistances=[1, 1],
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
