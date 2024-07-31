import numpy as np
import pytest

from CADETPythonSimulator.residual import (
    calculate_residual_volume_cstr,
    calculate_residual_concentration_cstr,
    calculate_residual_cake_vol_def,
    calculate_residual_press_easy_def
)
from CADETPythonSimulator.exception import CADETPythonSimError



# random number test
TestCaseCSTRConc_level1 = {
    "values" : {
        "c" : np.array([1, 2, 3]),
        "c_dot" : np.array([4, 5, 6]),
        "V" : 1,
        "V_dot" : 2,
        "Q_in" : 3,
        "Q_out" : 4,
        "c_in" : np.array([7, 8, 9])
    },
    "expected": np.array([-11, -7, -3])
}

# flow in and out are equal, concentrations to
TestCaseCSTRConc_equal = {
    "values" : {
        "c" : np.array([0.1,]),
        "c_dot" : np.array([0,]),
        "V" : 1,
        "V_dot" : 0,
        "Q_in" : 1,
        "Q_out" : 1,
        "c_in" : np.array([0.1,])
    },
    "expected": np.array([0,])
}

# flow in and out are equal, but concentrations going into the unit is not
TestCaseCSTRConc_diffcin = {
    "values" : {
        "c" : np.array([0.1,]),
        "c_dot" : np.array([0,]),
        "V" : 1.0,
        "V_dot" : 0.0,
        "Q_in" : 1.0,
        "Q_out" : 1.0,
        "c_in" : np.array([0.2,])
    },
    "expected": np.array([-0.1,])
}

#flow in and out are not equal, concentrantions going in are
TestCaseCSTRConc_diffvol = {
    "values" : {
        "c" : np.array([0.1,]),
        "c_dot" : np.array([0,]),
        "V" : 1.0,
        "V_dot" : 1.0,
        "Q_in" : 2.0,
        "Q_out" : 1.0,
        "c_in" : np.array([0.1,])
    },
    "expected": np.array([0,])
}

#flow in and out are not, equal, concentrations aren't equal too
TestCaseCSTRConc_diffvolc = {
    "values" : {
        "c" : np.array([0.1,]),
        "c_dot" : np.array([0.2,]),
        "V" : 1.0,
        "V_dot" : 1.0,
        "Q_in" : 2.0,
        "Q_out" : 1.0,
        "c_in" : np.array([0.2,])
    },
    "expected": np.array([0,])
}


@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseCSTRConc_level1,
        TestCaseCSTRConc_equal,
        TestCaseCSTRConc_diffcin,
        TestCaseCSTRConc_diffvol,
        TestCaseCSTRConc_diffvolc
    ]
)
class TestResidualConcCSTR():
    def test_calculation_concentration_cstr(self, parameters):

        param_vec_conc = parameters["values"].values()

        np.testing.assert_array_almost_equal(
            calculate_residual_concentration_cstr(*param_vec_conc),
            parameters["expected"]
            )

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

        np.testing.assert_equal(
            calculate_residual_volume_cstr(*param_vec_volume),
            parameters["expected"]
            )

# Testcase 1: Membrane rejects all
TestCaseDEFCake_rejects_all = {
    "values" : {
        "V_dot_f" : 1.0,
        "rejection" : np.array([1, 1]),
        "molar_volume" : np.array([1, 1]),
        "c_in" : np.array([0.5, 0.5]),
        "V_dot_C" : 1.0
    },
    "expected" : 0
}


# Testcase 2: Membrane rejects nothing
TestCaseDEFCake_rejects_not = {
    "values" : {
        "V_dot_f" : 1.0,
        "rejection" : np.array([0, 0]),
        "molar_volume" : np.array([1, 1]),
        "c_in" : np.array([0.5, 0.5]),
        "V_dot_C" : 0.0
    },
    "expected" : 0
}

# Testcase 3: Membrane rejects only Component 2
TestCaseDEFCake_rejects_2 = {
    "values" : {
        "V_dot_f" : 1.0,
        "rejection" : np.array([0, 1]),
        "molar_volume" : np.array([1, 1]),
        "c_in" : np.array([0.5, 0.5]),
        "V_dot_C" : 0.5
    },
    "expected" : 0
}

# Testcase 4: Component 2 is larger then 1
TestCaseDEFCake_C2_le_C1 = {
    "values" : {
        "V_dot_f" : 1.0,
        "rejection" : np.array([1, 1]),
        "molar_volume" : np.array([0.5, 1]),
        "c_in" : np.array([0.5, 0.5]),
        "V_dot_C" : 0.75
    },
    "expected" : 0
}

@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseDEFCake_rejects_all,
        TestCaseDEFCake_rejects_not,
        TestCaseDEFCake_rejects_2,
        TestCaseDEFCake_C2_le_C1
    ]
)

class TestResidualCakeVolDEF():
    def test_calculation_def(self, parameters):

        param_vec_cake_vol = parameters["values"].values()

        np.testing.assert_equal(calculate_residual_cake_vol_def(*param_vec_cake_vol),
            parameters["expected"]
            )

# Case 1 : Equally large hyraulic resistance
TestCaseDEFPressureDrop = {
    "values" : {
        "V_dot_P" : 1,
        "V_C" : 1,
        "deltap" : 0.5,
        "A" : 1,
        "mu" : 1,
        "Rm" : 1,
        "alpha" : 1,
    },
    "expected" : 0
}

# Case 2 : No cake yet
TestCaseDEFPressureDrop_no_cake = {
    "values" : {
        "V_dot_P" : 0.5,
        "V_C" : 0,
        "deltap" : 0.5,
        "A" : 1,
        "mu" : 1,
        "Rm" : 1,
        "alpha" : 1,
    },
    "expected" : 0
}

@pytest.mark.parametrize(
    "parameters",
    [
        TestCaseDEFPressureDrop,
        TestCaseDEFPressureDrop_no_cake
    ]
)


class TestResidualPressureDropDEF():
    def test_calculation_def(self, parameters):

        param_vec_pressure = parameters["values"].values()

        np.testing.assert_equal(calculate_residual_press_easy_def(*param_vec_pressure),
            parameters["expected"]
            )


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

