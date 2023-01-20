"""Provide tests for the functions that build up the synthetic control
method.
"""

import pytest
import sys
import numpy as np
import src.model_code.synth_control_functions as functions


# Tests for find_sunny_donors(x_tilde):
def test_normal_case_three_sunny_donors():
    argument = np.array([[-1., 0., 1.], [-1., -1., -1.]])
    sunny_donors, number, ids_sunny, ids_shady = functions.find_sunny_donors(
            argument)
    assert np.array_equal(sunny_donors,
                          np.array([[-1., 0., 1.], [-1., -1., -1.]])) \
        and number == 3 and ids_sunny == [0, 1, 2] and ids_shady == []


def test_no_sunny_donors():
    argument = np.array([[-1., 0., 1., 2.], [-1., -1., -1., 2.]])
    sunny_donors, number, ids_sunny, ids_shady = functions.find_sunny_donors(
            argument)
    assert np.array_equal(sunny_donors, np.array([[]])) and number == 0 \
        and ids_sunny == [] and ids_shady == [0, 1, 2, 3]


def test_three_out_of_five_sunny_donors():
    argument = np.array([[1., 3., 2., 2.4, 2.6], [3., 1., 2., 1.4, 1.6]])
    sunny_donors, number, ids_sunny, ids_shady = functions.find_sunny_donors(
            argument)
    assert np.array_equal(sunny_donors, np.array(
                          [[1., 3., 2.4], [3., 1., 1.4]])) \
        and number == 3 and ids_sunny == [0, 1, 3] and ids_shady == [2, 4]


def test_wrong_input_sunny_donors():
    argument = np.array([[[1., 3., 2., 2.4, 2.6], [3., 1., 2., 1.4, 1.6]]])
    with pytest.raises(ValueError):
        functions.find_sunny_donors(argument)


# Tests for inner_optimization_function(w, x_tilde, v):
def test_normal_case_inner_optimization_function():
    x_tilde = np.array([[-1., 0., 1.], [-1., -1., -1.]])
    v = np.diag([1., 1.])
    w = np.array([1.0, 1., 1.])
    assert functions.inner_optimization_function(w=w, x_tilde=x_tilde,
                                                 v=v) == 9.0


def test_normal_symmetric_case_inner_optimization_function():
    x_tilde = np.array([[1., 2., 3.], [1., 1., 3.], [2., 2., 3.]])
    v = np.diag([1., 1., 0.])
    w = np.array([0., 1., 2.])
    assert functions.inner_optimization_function(w=w, x_tilde=x_tilde,
                                                 v=v) == 113.


def test_wrong_input_inner_optimization_function():
    x_tilde = np.array([[1., 2., 3.], [1., 1., 3.], [2., 2., 3.]])
    v = np.diag([1., 1., 0.])
    w = np.array([0., 1.])
    with pytest.raises(ValueError):
        functions.inner_optimization_function(w=w, x_tilde=x_tilde, v=v)

# Tests of inner optimization with minimize (some kind of integration testing,
# uses inner optimization function)
def test_inner_optimization_one_optimal_column():
    x_tilde = np.array([[1., 2., 3.], [1., 1., 3.], [2., 2., 3.]])
    v = np.diag([1., 1., 1.])
    assert np.allclose(functions.inner_optimization(x_tilde, v),
                       np.array([1., 0., 0.]), rtol=1e-05, atol=1e-8)


def test_inner_optimization_two_optimal_columns():
    x_tilde = np.array([[1., -1., 5.], [2., -2., 5.], [1., -1., 5.]])
    v = np.diag([1., 1., 1.])
    assert np.allclose(functions.inner_optimization(x_tilde, v),
                       np.array([0.5, 0.5, 0.]), rtol=1e-05, atol=1e-8)


def test_inner_optimization_three_optimal_columns():
    x_tilde = np.array([[1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]])
    v = np.diag([1., 1., 1.])
    assert np.allclose(functions.inner_optimization(x_tilde, v),
                       np.array([1/3, 1/3, 1/3]), rtol=1e-05, atol=1e-8)


def test_wrong_input_inner_optimization():
    x_tilde = np.array([[1., 1., -1.], [1., -1., 1.]])
    v = np.diag([1., 1., 1.])
    with pytest.raises(ValueError):
        functions.inner_optimization(x_tilde, v)


# Integration tests for outer optimization function.
def test_outer_optimization_simple_case_weights_with_two_optimal_columns():
    v_k_tilde = np.array([0., 0.])
    x_tilde = np.array([[1., -1., 5.], [2., -2., 5.], [1., -1., 5.]])
    z_tilde = np.array([[1., 2., 1.], [0., 1., 3.], [3., 1., 2.]])
    assert np.allclose(functions.outer_optimization_function(
            v_k_tilde=v_k_tilde, x_tilde=x_tilde, z_tilde=z_tilde, k=2, K=3),
        6.5, rtol=1e-05, atol=1e-8)


def test_outer_optimization_other_v_weights_with_one_optimal_column():
    v_k_tilde = np.array([np.log10(0.5), np.log10(0.5)])
    x_tilde = np.array([[1., 2., 3.], [1., 1., 3.], [2., 2., 3.]])
    z_tilde = np.array([[1., 2., 1.], [0., 1., 3.], [3., 1., 2.]])
    assert np.allclose(functions.outer_optimization_function(
            v_k_tilde=v_k_tilde, x_tilde=x_tilde, z_tilde=z_tilde, k=2, K=3),
        10., rtol=1e-05, atol=1e-8)


# Test building of v from v_k_tilde.
def test_create_matrix_v_from_v_k_tilde():
    v_k_tilde = np.array([np.log10(0.1), np.log10(0.2), np.log10(1.)])
    k = 2
    assert np.allclose(functions.create_matrix_v_from_v_k_tilde(
            v_k_tilde=v_k_tilde, k=k),
        np.array([[0.1, 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0.2, 0.],
                 [0., 0., 0., 1.]]), rtol=1e-05, atol=1e-8)


# Test function used for outer optimization.
def test_normal_case_function_outer_optimization():
    z = np.array([[1., 3., 1.], [2., 2., 1.], [3., 1., 3.]])
    w = np.array([1., 1., 1.])
    assert functions.function_to_solve_for_outer_optimum(w=w, z_tilde=z) == \
        99.0


def test_only_one_weight_function_outer_optimization():
    z = np.array([[1., 3., 1.], [2., 2., 1.], [3., 1., 3.]])
    w = np.array([1., 0., 0.])
    assert functions.function_to_solve_for_outer_optimum(w=w, z_tilde=z) == \
        14.0


# Integration test of function to find unrestricted outer optimum.
def test_two_positive_weights_unrestricted_outer_optimum():
    z = np.array([[-1., 1., 0.], [1., -1., 2.], [0., 0., 1.]])
    assert np.allclose(functions.determine_unrestricted_outer_optimum(z),
                       np.array([0.5, 0.5, 0]), rtol=1e-05, atol=1e-8)


def test_three_positive_weights_unrestricted_outer_optimum():
    z = np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
    assert np.allclose(functions.determine_unrestricted_outer_optimum(z),
                       np.array([1/3, 1/3, 1/3]), rtol=1e-05, atol=1e-8)


# Test the function to build up the matrix for inequ constr of the linear
# program in (16).
def test_normal_case_build_up_a_ub():
    x = np.array([[1., 2., 3.], [2., 1., 0.]])
    w = np.array([1., 2., 1.])
    assert np.array_equal(
            functions.calculate_inner_part_of_inequ_constr_for_linprog_16(
                    x_tilde=x, w_outer=w), np.array([[56., 8.], [48., 12.],
                                                    [40., 16.]]))


def test_only_one_weight_build_up_a_ub():
    x = np.array([[1., 2], [2., 1.], [3., 0.]])
    w = np.array([1., 0.])
    assert np.array_equal(
            functions.calculate_inner_part_of_inequ_constr_for_linprog_16(
                    x_tilde=x, w_outer=w), np.array([[0., 0., 0.],
                                                    [-1., 2., 9.]]))


# Integration test for testing the function to determine if the outer optimum
# is feasible.
def test_try_if_outer_optimum_feasible_with_feasible_solution():
    x = np.array([[1., 2.], [2., 1.], [3., 0.]])
    z = np.array([[-1., -2.], [-2., -2.], [1., 2.]])
    status, w_optimal, v_optimal = \
        functions.try_if_unrestricted_outer_optimum_feasible(x, z)
    assert status == "Outer optimum feasible." \
        and np.allclose(w_optimal, np.array([1., 0.]), rtol=1e-05, atol=1e-8) \
        and np.allclose(v_optimal, np.array(
            [[1, 0., 0.], [0., 10**(-8)/(11*10**(-8)), 0.],
             [0., 0., 10**(-8)/(11*10**(-8))]]),
                rtol=1e-05, atol=1e-8)


def test_try_if_outer_optimum_feasible_with_infeasible_solution():
    x = np.array([[1., 2., 4.], [2., 1., 4.]])
    z = np.array([[-1., 1., 0.], [2., -2., 0.], [-5., -5., 0.]])
    status, w_optimal, v_optimal = \
        functions.try_if_unrestricted_outer_optimum_feasible(x, z)
    assert status == "Outer optimum is infeasible." \
        and np.isnan(w_optimal) and np.isnan(v_optimal)


# Test case of no sunny donors, just perform inner minimization and return
# warning, integration tast.
def test_solve_case_of_no_sunny_donors_normal_case():
    x = np.array([[1., -1., -1.], [0., 0., 0.], [1., -1., -1.]])
    status, w_optimal, v_optimal = functions.solve_case_of_no_sunny_donors(x)
    assert status == """No sunny donors, perfect fit of treated unit regarding
    predictors possible. WARNING: In this case the outer optimization is not
    performed. This means The solution might not choose the optimal weights
    concerning the pretreatment fit. To get rid of this problem choose more
    variables  defining the pre-treatment fit as predictors.""" \
        and sum(w_optimal) == 1. and w_optimal.shape == (3,) and \
        np.array_equal(v_optimal, np.diag([1., 1., 1.]))


# System test of whole functions part using the function to determine optimal
# synthetic control weights.
def test_determine_synthetic_control_weights_no_sunny_donors():
    x = np.array([[1., -1., -1.], [0., 0., 0.], [1., -1., -1.]])
    z = np.array([[1., 3., -1.], [0., 1., 0.], [1, 3., -1.]])
    status, weights, v, ids_sunny_donors = \
        functions.determine_synthetic_control_weights(x_tilde=x, z_tilde=z)
    assert status == """No sunny donors, perfect fit of treated unit regarding
    predictors possible. WARNING: In this case the outer optimization is not
    performed. This means The solution might not choose the optimal weights
    concerning the pretreatment fit. To get rid of this problem choose more
    variables  defining the pre-treatment fit as predictors.""" \
        and sum(weights) == 1. and weights.shape == (3,) and \
        np.array_equal(v, np.diag([1., 1., 1.]))


def test_determine_synthetic_control_weights_one_sunny_donor():
    x = np.array([[1., 2., 3.], [1., 2., 3.]])
    z = np.array([[1., 3., -1.], [0., 1., 0.], [1, 3., -1.]])
    status, weights, v, ids_sunny_donors = \
        functions.determine_synthetic_control_weights(x_tilde=x, z_tilde=z)
    assert status == "Only one sunny donor, this donor gets maximum weight." \
        and np.array_equal(weights, np.array([1., 0., 0.])) and \
        np.array_equal(v, np.diag([1., 1.]))


def test_determine_synthetic_control_weights_outer_optimum_feasible():
    x = np.array([[1., -2., -2.], [0., 1., 1.], [1., -2., -2.]])
    z = np.array([[-2., 1., 4.], [-4., 2., 8.], [-2, 3., 4.]])
    status, weights, v, ids_sunny_donors = \
        functions.determine_synthetic_control_weights(x_tilde=x, z_tilde=z)
    assert status == \
        "Outer optimum feasible, weights of outer optimum chosen." and \
        np.allclose(weights, np.array([2/3., 0., 1/3]), rtol=1e-3, atol=1e-5) \
        and np.amax(v) == 1. and v.shape == (3, 3)


def test_determine_synthetic_control_weights_perform_nested_optimization():
    x = np.array([[-1., 2., 1.], [-2., 3., 2.], [1., 2., 1.]])
    z = np.array([[-1., 1., 2.], [-2., 2., 3.], [1, -1., 0.]])
    status, weights, v, ids_sunny_donors = \
        functions.determine_synthetic_control_weights(x_tilde=x, z_tilde=z)
    assert status == "Nested optimization was performed successfully." \
        and np.allclose(weights, np.array([0.5, 0., 0.5]),
                        rtol=1e-3, atol=1e-5) and np.amax(v) == 1. and \
        v.shape == (3, 3)


if __name__ == '__main__':
    status = pytest.main([sys.argv[1]])
    sys.exit(status)
