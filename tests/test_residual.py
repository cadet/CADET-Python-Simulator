import numpy as np
import pytest

from CADETPythonSimulator.residual import (
    calculate_residual_volume_cstr, calculate_residual_concentration_cstr
)
from CADETPythonSimulator.exception import CADETPythonSimError


# random number test
TestCaseConc_level1 = {
    "values": {
        "c": np.array([1, 2, 3]),
        "c_dot": np.array([4, 5, 6]),
        "V": 1,
        "V_dot": 2,
        "Q_in": 3,
        "Q_out": 4,
        "c_in": np.array([7, 8, 9])
    },
    "expected": np.array([-11, -7, -3])
}

# flow in and out are equal, concentrations to
TestCaseConc_equal = {
    "values": {
        "c": np.array([0.1,]),
        "c_dot": np.array([0,]),
        "V": 1,
        "V_dot": 0,
        "Q_in": 1,
        "Q_out": 1,
        "c_in": np.array([0.1,])
    },
    "expected": np.array([0,])
}

# flow in and out are equal, but concentrations going into the unit are not
TestCaseConc_diffcin = {
    "values": {
        "c": np.array([0.1,]),
        "c_dot": np.array([0,]),
        "V": 1.0,
        "V_dot": 0.0,
        "Q_in": 1.0,
        "Q_out": 1.0,
        "c_in": np.array([0.2,])
    },
    "expected": np.array([-0.1,])
}

# flow in and out are not equal, concentrantions going in are
TestCaseConc_diffvol = {
    "values": {
        "c": np.array([0.1,]),
        "c_dot": np.array([0,]),
        "V": 1.0,
        "V_dot": 1.0,
        "Q_in": 2.0,
        "Q_out": 1.0,
        "c_in": np.array([0.1,])
    },
    "expected": np.array([0,])
}

# flow in and out are not, equal, concentrations aren't equal too
TestCaseConc_diffvolc = {
    "values": {
        "c": np.array([0.1,]),
        "c_dot": np.array([0.2,]),
        "V": 1.0,
        "V_dot": 1.0,
        "Q_in": 2.0,
        "Q_out": 1.0,
        "c_in": np.array([0.2,])
    },
    "expected": np.array([0,])
}


@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseConc_level1,
        TestCaseConc_equal,
        TestCaseConc_diffcin,
        TestCaseConc_diffvol,
        TestCaseConc_diffvolc
    ]
)
class TestResidualConcCSTR():
    def test_calculation_concentration_cstr(self, parameters):

        param_vec_conc = parameters["values"].values()

        residual = calculate_residual_concentration_cstr(*param_vec_conc)

        np.testing.assert_array_almost_equal(residual, parameters["expected"])


# random number test
TestCaseVol = {
    "values": {
        "V": 1,
        "V_dot": 2,
        "Q_in": 3,
        "Q_out": 4,
    },
    "expected": 3
}

# Flow in and out are equal
TestCaseVol_equal = {
    "values": {
        "V": 1,
        "V_dot": 0,
        "Q_in": 1,
        "Q_out": 1,
    },
    "expected": 0
}

# Flow in is larger than out
TestCaseVol_inge = {
    "values": {
        "V": 1,
        "V_dot": 1,
        "Q_in": 2,
        "Q_out": 1,
    },
    "expected": 0
}

# Flow in is lesser than out
TestCaseVol_inle = {
    "values": {
        "V": 1,
        "V_dot": -1,
        "Q_in": 1,
        "Q_out": 2,
    },
    "expected": 0
}

# Residual does not depend on Volumne

TestCaseVol_vol = {
    "values": {
        "V": 1e10,
        "V_dot": 0,
        "Q_in": 0,
        "Q_out": 0,
    },
    "expected": 0
}


@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseVol,
        TestCaseVol_equal,
        TestCaseVol_inge,
        TestCaseVol_inle,
        TestCaseVol_vol
    ]
)
class TestResidualVolCSTR():
    def test_calculation_cstr(self, parameters):

        param_vec_volume = parameters["values"].values()

        residual = calculate_residual_volume_cstr(*param_vec_volume)

        np.testing.assert_equal(residual, parameters["expected"])


TestCaseConcError = {
    "values": {
        "c": np.array([1, 2, 3]),
        "c_dot": np.array([4, 5, 6]),
        "V": -1,
        "V_dot": 2,
        "Q_in": 3,
        "Q_out": 4,
        "c_in": np.array([7, 8, 9])
    },
    "expected": np.array([-11, -7, -3])
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
