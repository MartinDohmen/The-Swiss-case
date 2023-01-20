"""Provide functions to calculate optimal weights for synthetic control unit.

"""

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize


def find_sunny_donors(x_tilde):
    """Find sunny donors from a numpy array of all donors according to (9)
    in Becker and Klößner (2018). Return a numpy array only including the data
    remaining for sunny donors and the number of sunny donors.
    Additionaly return the index in the matrix of columns for sunny and shady
    donors."""

    # Define array to save all values for sunny donors and a list for
    # the ids of donors.
    sunny_donors = np.array([[]])
    counter_column = 0
    ids_sunny_donors = []
    ids_shady_donors = []

    # Build matrix structures necessary for linear program from (9).
    c = np.concatenate((np.array([1.]), np.zeros(x_tilde.shape[1])), axis=0)
    b_eq = np.concatenate((np.zeros(x_tilde.shape[0]), np.array([1.])), axis=0)
    A_eq_rest = np.concatenate((-x_tilde,
                                np.array([np.ones(x_tilde.shape[1])])), axis=0)

    for column in x_tilde.T:
        A_eq_first_column = np.array([np.concatenate((column,
                                                      np.zeros(1)), axis=0)])
        A_eq = np.concatenate((A_eq_first_column.T, A_eq_rest), axis=1)

        # Run linear program from (9).
        sol_linear_prog = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                                  options={'disp': False, 'bland': False,
                                           'tol': 1e-12, 'maxiter': 50000})

        assert sol_linear_prog['success'], \
            """Linear program to determine if a donor is sunny from (9) was
            not successfull for column {} of x_tilde, because of {}
            """.format(column, sol_linear_prog['message'])

        # Check if column belongs to a sunny donor.
        if np.isclose(sol_linear_prog['fun'], 1., rtol=1e-05, atol=1e-08):
            # Save sunny donors.
            new_donor = np.array([column]).T
            if sunny_donors.size == 0:
                sunny_donors = new_donor
            else:
                sunny_donors = np.column_stack((sunny_donors, new_donor))

            # Save ids of donor for sunny or shady.
            ids_sunny_donors.append(counter_column)
        else:
            ids_shady_donors.append(counter_column)
        counter_column = counter_column + 1

    # Get the number of sunny donors.
    number_sunny_donors = sunny_donors.shape[1]

    return sunny_donors, number_sunny_donors, ids_sunny_donors, \
        ids_shady_donors


def inner_optimization(x_tilde, v):
    """Perform the inner optimization of the synthetic control problem,
    corresponding to equation (8) in Becker and Klößner (2018).
    Return optimal weights for countries w."""

    # Define start vector for weights.
    w = np.full(x_tilde.shape[1], 1/x_tilde.shape[1])

    # Define the bounds for the optimization problem.
    bounds = []
    for i in range(0, x_tilde.shape[1]):
        bounds.append((0, 1))

    # Define constraints.
    constraints = {'type': 'eq',
                   'fun': lambda x: np.ones(x_tilde.shape[1]) @ x.T - 1.}

    # Perform the minimization.
    inner_minimization_results = minimize(fun=inner_optimization_function,
                                          x0=w,
                                          args=(x_tilde, v),
                                          method='SLSQP',
                                          bounds=bounds,
                                          constraints=constraints
                                          )

    assert inner_minimization_results['success'], \
        """Inner optimization was not successfull for v = {}, \n because
        of {}""".format(
        v, inner_minimization_results['message'])

    return inner_minimization_results['x']


def inner_optimization_function(w, x_tilde, v):
    """Construct the function to minimize by the inner optimization, 
    corresponding to the part after min in (8) in Becker and Klößner (2018)"""

    return w @ x_tilde.T @ v @ x_tilde @ w.T


def outer_optimization(z_tilde, x_tilde, lower_bound=10**-8):
    """Perform the outer optimization of equation (7) in Becker and klößner
    (2018). Use K optimizations of dimension K-1 as described in the paper
    in 3.5.. Return the optimal constrains weighting matrix v."""

    # Determine K, the number of predictors.
    K = x_tilde.shape[0]

    # Define start value.
    v_k_tilde_start = np.zeros(K-1)

    # Define the bounds for the outer optimization problem.
    bounds = []
    for i in range(0, K-1):
        bounds.append((np.log10(lower_bound), 0))

    # Solve the K optimization problems and take the solution of the one
    # leading to the smallest function value.
    for k in range(1, K+1):
        outer_minimization_results = minimize(fun=outer_optimization_function,
                                              x0=v_k_tilde_start,
                                              args=(z_tilde, x_tilde, k, K),
                                              method='L-BFGS-B',
                                              bounds=bounds
                                              )

        assert outer_minimization_results['success'], \
            """Outer optimization was not successfull for k = {}, because of {}
            """.format(k, outer_minimization_results['message'])

        optimal_v_k_tilde = outer_minimization_results['x']

        if k == 1:
            optimal_v = create_matrix_v_from_v_k_tilde(optimal_v_k_tilde, k)
            optimal_function_value = outer_minimization_results['fun']
        elif outer_minimization_results['fun'] < optimal_function_value:
            optimal_v = create_matrix_v_from_v_k_tilde(optimal_v_k_tilde, k)
            optimal_function_value = outer_minimization_results['fun']

    return optimal_v


def outer_optimization_function(v_k_tilde, z_tilde, x_tilde, k, K):
    """Construct the function to minimize in the K subproblems of the outer
    optimization. For that consruct V according to (20) and than construct
    function used in (7) in Becker and klößner (2018)."""

    # Create matrix V from v_k_tilde.
    v = create_matrix_v_from_v_k_tilde(v_k_tilde, k)

    # Perform inner optimization to get optimal weights from v.
    w_star_of_v = inner_optimization(x_tilde, v)

    # Solve equation to minimize in (7).
    solution_inner_function_outer_optimum = \
        function_to_solve_for_outer_optimum(w_star_of_v, z_tilde)

    return solution_inner_function_outer_optimum


def ten_to_the_power_of_x(x):
    """Calculate 10 to the power of x."""

    return 10 ** x


def create_matrix_v_from_v_k_tilde(v_k_tilde, k):
    """Create matrix V from v_k_tilde about which to optimize, see (20)
    in Becker and Klößner (2018)."""

    ten_to_the_power_of_x_vectors = np.vectorize(ten_to_the_power_of_x)
    v_k_minus_one = ten_to_the_power_of_x_vectors(v_k_tilde)
    v_k = np.insert(v_k_minus_one, k-1, 1)
    v = np.diag(v_k)

    return v


def function_to_solve_for_outer_optimum(w, z_tilde):
    """Function to solve for the outer optimum as used in (7) or (13) of
    Becker and Klößner (2018)."""

    return w @ z_tilde.T @ z_tilde @ w.T


def determine_unrestricted_outer_optimum(z_tilde):
    """Solve euqtion (13) in Becker and Klößner (2018) to get weights solving
    the unrestricted outer optimization problem. Return these weights."""

    # Define start vector for weights.
    w_start = np.full(z_tilde.shape[1], 1/z_tilde.shape[1])

    # Define the bounds for the inner optimization problem.
    bounds = []
    for i in range(0, z_tilde.shape[1]):
        bounds.append((0, 1))

    # Define constraints.
    constraints = {'type': 'eq',
                   'fun': lambda x: np.ones(z_tilde.shape[1]) @ x.T - 1.}

    # Perform the minimization.
    unrestricted_outer_minimization_results = minimize(
            fun=function_to_solve_for_outer_optimum,
            x0=w_start,
            args=(z_tilde),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
            )
    assert unrestricted_outer_minimization_results['success'], \
        "Unrestricet outer optimization was not successfull."

    return unrestricted_outer_minimization_results['x']


def try_if_unrestricted_outer_optimum_feasible(x_tilde, z_tilde,
                                               lower_bound=10**-8):
    """Solve linear program in (16) in Becker and Klößner (2018).
    This means, determine if the outer optimum is feasible.
    If so, return a status telling that and the weights forming the outer
    optimum as well as the normalized constraint weighting matrix V belonging
    to the outer optimum. If not, return a status telling the outer optimum is
    infeasible and give numpy.nan as weights w and V.
    """

    w_outer = determine_unrestricted_outer_optimum(z_tilde)

    # Define coefficients for the linear program.
    c = np.ones(x_tilde.shape[0])
    A_ub = calculate_inner_part_of_inequ_constr_for_linprog_16(x_tilde,
                                                               w_outer)
    b_ub = np.zeros(w_outer.shape)

    # Define bounds for the linear program.
    bounds = (lower_bound, 1.)

    # Run the linear program from (16).
    solution_linear_prog = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    # Save variables to return if unrestricted outer optimum feasible.
    if solution_linear_prog['success']:
        w_optimal = w_outer
        v_optimal = np.diag(solution_linear_prog['x'])
        status = "Outer optimum feasible."

        # Normalize v_optimal.
        max_value_of_v = np.amax(v_optimal)
        v_optimal = v_optimal * (1/max_value_of_v)

    # Save variables if unrestricted outer optimum not feasible: nan shows that
    # the optimal values have not been found yet.
    else:
        w_optimal = np.nan
        v_optimal = np.nan
        status = "Outer optimum is infeasible."

    return status, w_optimal, v_optimal


def calculate_inner_part_of_inequ_constr_for_linprog_16(
        x_tilde, w_outer):
    """Calculate the inner part of the inequality constraints for the linear +
    program in (20) from Becker and Klößner (2018). This is, calculate
    (e_j - v) ' B_k  w for k=1,...,K and stack them into a vector. Do that for
    all countries J, so j = 1,..., J and stack them into a matrix to get the
    matrix defining the inequality constraint of the linear program A_ub."""

    K = x_tilde.shape[0]
    J = x_tilde.shape[1]

    # Create matrix -A_ub to save calculated arrays.
    minus_a_ub = np.array([[]])

    for j in range(1, J+1):
        # Create unit vector of the right shape (e_j).
        e_j = np.zeros(J)
        e_j[j-1] = 1.0

        # Create vector to save inner blog of inequ. constr. and to build up
        # A_ub.
        vector_inner_blog = np.array([])

        for k in range(1, K+1):
            # Create diagonal matrix V(0,..,0,1,0...,0) to calculate B_k.
            e_k = np.zeros(K)
            e_k[k-1] = 1.0
            v = np.diag(e_k)

            # Calculate B_k.
            b_k = x_tilde.T @ v @ x_tilde

            # Calculate inner blog of inequ constr. inside the sum in (16).
            e_j_minus_w = e_j - w_outer
            inner_blog_inequ_constr = e_j_minus_w @ b_k @ w_outer
            vector_inner_blog = np.append(vector_inner_blog,
                                          inner_blog_inequ_constr)

        # Append the vector for the inner blog to build up matrix a_ub.
        matrix_repr_vector_inner_blog = np.array([vector_inner_blog])
        if j == 1:
            # Create matrix -A_ub to save calculated arrays.
            minus_a_ub = matrix_repr_vector_inner_blog
        else:
            minus_a_ub = np.vstack((minus_a_ub, matrix_repr_vector_inner_blog))

    a_ub = - minus_a_ub
    return a_ub


def solve_case_of_no_sunny_donors(x_tilde):
    """Solve the case when we have no sunny donors, so a perfect fit of the
    treated unit is possible. In this case I deviate from the paper by
    Becker and Klößner (2018). In their paper they solve the linear programm
    (10). This does not work with scipy.optimize as the minimizer does not
    support problems with more equality constraints than independent variables.
    As I did not find a good optimizer for python so far solving this problem,
    I will just choose a weights solving the inner minimization, so leading to
    a perfect fit for the synthetic control unit regarding the predictor
    variables. This might not be optimal concerning the outer optimization!
    The reason for this shortcut is that the case of perfect fit is very
    unprobable to begin with. If this case happens, the program will print a
    warning. In this case a good solution would be to choose more predictor
    variables. In an extrem case all variables/data points defining the outer
    optimization can be used as predictors. This would lead to the problem being
    unimportant as outer and inner optimization become the same!
    """

    # V arbitrary choice, here equal weights.
    v_optimal_vector = np.ones(x_tilde.shape[0])
    v_optimal = np.diag(v_optimal_vector)

    # Perform only the inner optimization.
    w_optimal = inner_optimization(x_tilde=x_tilde, v=v_optimal)

    status = """No sunny donors, perfect fit of treated unit regarding
    predictors possible. WARNING: In this case the outer optimization is not
    performed. This means The solution might not choose the optimal weights
    concerning the pretreatment fit. To get rid of this problem choose more
    variables  defining the pre-treatment fit as predictors."""

    return status, w_optimal, v_optimal


def determine_synthetic_control_weights(x_tilde, z_tilde, lower_bound=10**-8):
    """Put everything together to determine the
    optimal country weights w and the optimal constraint weighting matrix v
    defining the synthetic control. As inputs use only numpy matrixes!
    The function works as described in the paper by Becker and Klößner (2018),
    especially it uses the algorithm shown in figure 2 of the paper. The
    difference to the paper so far is that it does not allow to pass a mapping
    V, but builds up V as a diagonal matrix of constraint weights. This means,
    so far it does not allow for time series as constraints with stable weights
    for the whole time series. Instead it always calculates for every single
    constraint the optimal weight to solve the outer minimization problem.
    This means do:
        
        1. determine sunny donors
        
        2. test of no sunny donors, if so solve outer minimization constraint
        to exact fit and stop.
        
        3. test of one sunny donor, if so just give him a positive weight
        and stop.
        
        4. try if unrestricted outer minimization feasible, if so choose
        weights solving unrestricted outer minimization and stop.
        
        5. Perform the nested optimization task and choose weights solving it.
    """

    # Determine sunny donors.
    x_tilde_sunny, number_sunny_donors, ids_sunny_donors, ids_shady_donors = \
        find_sunny_donors(x_tilde=x_tilde)

    # Case of no sunny donor, perfect fit possible.
    if number_sunny_donors == 0:
        status, w_optimal, v_optimal = solve_case_of_no_sunny_donors(x_tilde)

    # Case of only one sunny donor, only he gets positive weight.
    elif number_sunny_donors == 1:
        w_optimal = np.zeros(x_tilde.shape[1])
        w_optimal[ids_sunny_donors[0]] = 1.
        # V arbitrary choice, here equal weights.
        v_optimal_vector = np.ones(x_tilde.shape[0])
        v_optimal = np.diag(v_optimal_vector)
        status = "Only one sunny donor, this donor gets maximum weight."

    else:
        # Test if unrestricted outer optimum feasible.
        status_unrestr_outer_opt, w_unrestr_outer_opt, v_unrestr_outer_opt = \
            try_if_unrestricted_outer_optimum_feasible(x_tilde=x_tilde,
                                                       z_tilde=z_tilde,
                                                       lower_bound=lower_bound)

        # Case if unrestricted outer optimum feasible.
        if status_unrestr_outer_opt == "Outer optimum feasible.":
            w_optimal = w_unrestr_outer_opt
            v_optimal = v_unrestr_outer_opt
            status = "Outer optimum feasible, weights of outer optimum chosen."

        # Else perform nested optimization task.
        else:
            # Safe columns of z_tilde belonging to sunny donors.
            z_tilde_sunny = z_tilde[:, ids_sunny_donors]

            # Perform nested optimization to get optimal constraint weights V.
            v_optimal = outer_optimization(z_tilde=z_tilde_sunny,
                                           x_tilde=x_tilde_sunny)

            # Perform inner optimization with optimal constraint weights to get
            # optimal donor weights w for sunny donors.
            w_optimal_sunny = inner_optimization(x_tilde=x_tilde_sunny,
                                                 v=v_optimal)

            # Construct optimal weights w including weights of sunny donors and
            # 0 weights for shady donors.
            w_optimal = np.zeros(x_tilde.shape[1])
            for index, content in enumerate(ids_sunny_donors):
                w_optimal[content] = w_optimal_sunny[index]

            status = "Nested optimization was performed successfully."

    return status, w_optimal, v_optimal, ids_sunny_donors
