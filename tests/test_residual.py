from CADETPythonSimulator.residual import (
    calculate_residual_volume_cstr, calculate_residual_concentration_cstr
)
from CADETPythonSimulator.exception import CADETPythonSimError
import pytest
import numpy as np


TestCaseConc = {
    "values" : {
        "c" : np.array([1, 2, 3]),
        "c_dot" : np.array([4, 5, 6]),
        "V" : 1,
        "V_dot" : 2,
        "Q_in" : 3,
        "Q_out" : 4,
        "c_in" : np.array([7, 8, 9])
    },
    "expected" : np.array([-11,-7,-3])
}



@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseConc
    ]
)
class TestResidualConcCSTR():
    def test_calculation_concentration_cstr(self, parameters):

        param_vec_conc = parameters["values"].values()

        np.testing.assert_equal(calculate_residual_concentration_cstr(*param_vec_conc), parameters["expected"])





TestCaseVol = {
    "values" : {
        "V" : 1,
        "V_dot" : 2,
        "Q_in" : 3,
        "Q_out" : 4,
    },
    "expected" : 3
}



@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseVol
    ]
)

class TestResidualVolCSTR():
    def test_calculation_cstr(self, parameters):

        param_vec_volume = parameters["values"].values()

        np.testing.assert_equal(calculate_residual_volume_cstr(*param_vec_volume), parameters["expected"])

TestCaseConcError = {
    "values" : {
        "c" : np.array([1, 2, 3]),
        "c_dot" : np.array([4, 5, 6]),
        "V" : -1,
        "V_dot" : 2,
        "Q_in" : 3,
        "Q_out" : 4,
        "c_in" : np.array([7, 8, 9])
    },
    "expected" : np.array([-11,-7,-3])
}


@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseConcError
    ]
)


class TestResidualError():

    def test_calculation_vol_cstr_error(self, parameters):

        param_vec_volume = parameters["values"].values()

        with pytest.raises(CADETPythonSimError):
            calculate_residual_volume_cstr(*list(param_vec_volume)[2:6])

    def test_calculation_concentration_cstr_error(self, parameters):

        param_vec_volume = parameters["values"].values()

        with pytest.raises(CADETPythonSimError):
            calculate_residual_concentration_cstr(*param_vec_volume)
