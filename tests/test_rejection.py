import numpy as np
import pytest

from CADETPythonSimulator.rejection import (
    RejectionBase, StepCutOff
)


TestCaseCutof = {
            "model": StepCutOff,
            "model_param": {
                "cutoff_weight": 0.5
            },
            "weights": [0, 0.5, 1],
            "expected": [0, 1, 1]
        }

@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseCutof
    ]
)
class TestRejection():
    """Test Class to test all rejection Models."""

    def test_get_rejection(self, parameters):
        """Test to check wheter the get_rejection function works as intended."""
        model = parameters["model"](**parameters["model_param"])

        solution = [model.get_rejection(weight) for weight in parameters["weights"]]
        np.testing.assert_array_almost_equal(solution, parameters["expected"])
