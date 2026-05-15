import numpy as np
import pytest
from CADETPythonSimulator.coupling_interface import WeightedAverageCoupling


class TestWeightedAverageCoupling:
    @pytest.fixture
    def coupling(self):
        return WeightedAverageCoupling()

    def test_single_source_returned_unchanged(self, coupling):
        origin_list = [({"c": np.array([1.0, 2.0])}, 1.0)]
        result = coupling.get_coupled_state(origin_list, "c")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_single_source_rate_cancels(self, coupling):
        origin_list = [({"c": np.array([3.0, 6.0])}, 2.0)]
        result = coupling.get_coupled_state(origin_list, "c")
        np.testing.assert_array_almost_equal(result, [3.0, 6.0])

    def test_two_equal_sources_equal_rates(self, coupling):
        origin_list = [
            ({"c": np.array([0.0, 0.0])}, 1.0),
            ({"c": np.array([2.0, 4.0])}, 1.0),
        ]
        result = coupling.get_coupled_state(origin_list, "c")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_weighted_average(self, coupling):
        origin_list = [
            ({"c": np.array([0.0, 0.0])}, 0.3e-3),
            ({"c": np.array([1.0, 1.0])}, 0.7e-3),
        ]
        result = coupling.get_coupled_state(origin_list, "c")
        np.testing.assert_array_almost_equal(result, [0.7, 0.7])

    def test_output_shape_matches_input(self, coupling):
        origin_list = [({"c": np.array([1.0, 2.0, 3.0])}, 1.0)]
        result = coupling.get_coupled_state(origin_list, "c")
        assert result.shape == (3,)

    def test_three_sources(self, coupling):
        origin_list = [
            ({"c": np.array([0.0])}, 1.0),
            ({"c": np.array([1.0])}, 1.0),
            ({"c": np.array([2.0])}, 1.0),
        ]
        result = coupling.get_coupled_state(origin_list, "c")
        np.testing.assert_array_almost_equal(result, [1.0])
