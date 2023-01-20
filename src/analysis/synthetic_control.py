"""Run the synthetic control method to define the weights for a synthetic
Switzerland.

All developed and tested using Jupyter notebooks.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj

from src.model_code.synth_control_functions import \
    determine_synthetic_control_weights


def standarize_data(data):
    """Standarize data to have mean zero and unit variance."""

    data_stand = (data - data.mean())/data.std()

    return data_stand


def get_z_tilde(dep_variable, index_treated, index_donors,
                index_pre_treatment_start, index_treatment_start):
    """Calculate Z_tilde = Z_0 - Z_1 * vector_ones', the input data for the
    outer minimization of the synthetic control method."""
    # Get Z_1, the pre-treatment data of the treated unit.
    z_one = get_z_one_from_data(dep_variable, index_treated,
                                index_pre_treatment_start,
                                index_treatment_start)

    # Get Z_0, the pre-treatment data of the control units.
    z_zero = get_z_zero_from_data(dep_variable, index_donors,
                                  index_pre_treatment_start,
                                  index_treatment_start)

    # Define z_tilde.
    z_one_times_vector_of_ones = z_one @ np.array([np.ones(z_zero.shape[1])])
    z_tilde = z_zero - z_one_times_vector_of_ones

    return z_tilde


def get_z_one_from_data(dep_variable, index_treated, index_pre_treatment_start,
                        index_treatment_start):
    """Calculate the array Z_one, the array of pre-treatment data of the
    dependent variable for the treated unit."""

    z_one = dep_variable.loc[index_treated].loc[
            index_pre_treatment_start:index_treatment_start-1]
    z_one_matrix = z_one.as_matrix(['Value'])

    return z_one_matrix


def get_z_zero_from_data(dep_variable, index_donors, index_pre_treatment_start,
                         index_treatment_start):
    """Calculate the array Z_one, the array of pre-treatment data of the
    dependent variable for the treated unit."""

    for index, donor_id in enumerate(index_donors):
        data_donor = dep_variable.loc[donor_id].loc[
                index_pre_treatment_start:index_treatment_start-1]
        if index == 0:
            z_zero = data_donor.as_matrix(['Value'])
        else:
            matrix_donor = data_donor.as_matrix(['Value'])
            z_zero = np.concatenate((z_zero, matrix_donor), axis=1)

    return z_zero


def get_x_tilde(dep_variable, predictor_data, predictors, index_treated,
                index_donors):
    """Calculate X_tilde = X_0 - X_1 * vector_ones', the input data for the
    inner minimization of the synthetic control method."""

    # Build up X_0 and X_1, the matrices containing the predictor variables.
    x_one, x_zero = get_x_one_and_x_zero_from_data(dep_variable,
                                                   predictor_data,
                                                   predictors,
                                                   index_treated, index_donors)

    # Define X_tilde.
    x_one_times_vector_of_ones = x_one @ np.array([np.ones(x_zero.shape[1])])
    x_tilde = x_zero - x_one_times_vector_of_ones

    return x_tilde


def get_x_one_and_x_zero_from_data(dep_variable, predictor_data, predictors,
                                   index_treated, index_donors):
    """Extraxt X_1 and x_0, the matrices of predictor values for treated unit
    and donors from the data. Build up matrices recursivly looping over
    predictor definitions."""

    for i, dictionary in enumerate(predictors):

        # Select data to get predictors from.
        if dictionary['data'] == 'dep_variable':
            data = dep_variable
        elif dictionary['data'] == 'predictor':
            data = predictor_data
        else:
            raise KeyError('Data type for construction of predictors not found'
                           )

        # Get values for X_0 and X_1 if data is from the dependent variable.
        if dictionary['type'] == 'average':
            sum_values_for_x_one = 0.

            # Get right entries of data and format the data points accordingly
            # for treated unit.
            for period in dictionary['periods']:
                period_index = pd.Index(periods).get_loc((period,))
                sum_values_for_x_one = sum_values_for_x_one + data.loc[
                        index_treated].loc[period_index]
            value_for_x_one = sum_values_for_x_one / len(dictionary['periods'])

            # Get right entries of data and format the data points accordingly
            # for donors.
            for index, donor_id in enumerate(index_donors):
                sum_values_for_x_zero = 0.
                for period in dictionary['periods']:
                    period_index = pd.Index(periods).get_loc((period,))
                    sum_values_for_x_zero = sum_values_for_x_zero + data.loc[
                            donor_id].loc[period_index]
                value_for_x_zero = sum_values_for_x_zero / len(dictionary[
                        'periods'])
                if index == 0:
                    row_of_x_zero = np.array([value_for_x_zero])
                else:
                    row_of_x_zero = np.concatenate((row_of_x_zero, np.array(
                            [value_for_x_zero])), axis=1)

        elif dictionary['type'] == 'point':

            # Get right entry of data for treated unit.
            period_index = pd.Index(periods).get_loc((dictionary['periods'],))
            value_for_x_one = data.loc[index_treated].loc[period_index]

            # Get right entries of data for donors.
            for index, donor_id in enumerate(index_donors):
                value_for_x_zero = data.loc[donor_id].loc[period_index]
                if index == 0:
                    row_of_x_zero = np.array([value_for_x_zero])
                else:
                    row_of_x_zero = np.concatenate((row_of_x_zero, np.array(
                            [value_for_x_zero])), axis=1)

        # Get values if data is from predictors instead of the dependent
        # variable.
        elif dictionary['data'] == 'predictor':

            assert dictionary['type'] == "trade_openness" or \
                dictionary['type'] == "industry_share" or\
                dictionary['type'] == "inflation_rate" or\
                dictionary['type'] == "schooling", \
                """Type of covariate as predictor not found."""

            # Get right entry of data for treated unit.
            value_for_x_one = np.array(
                    [data.loc[index_treated][dictionary['type']]])

            # Get right entries of data for donors.
            for index, donor_id in enumerate(index_donors):
                value_for_x_zero = np.array(
                        [data.loc[donor_id][dictionary['type']]])
                if index == 0:
                    row_of_x_zero = np.array([value_for_x_zero])
                else:
                    row_of_x_zero = np.concatenate((row_of_x_zero, np.array(
                            [value_for_x_zero])), axis=1)

        else:
            raise KeyError("""Type of conversion for dependent variable as
                           predictor not found""")

        # Save data for X_1.
        if i == 0:
            x_one = np.array([value_for_x_one])
        else:
            x_one = np.concatenate((x_one, np.array([value_for_x_one])),
                                   axis=0)

        # Save data for X_0.
        if i == 0:
            x_zero = row_of_x_zero
        else:
            x_zero = np.concatenate((x_zero, row_of_x_zero), axis=0)

    return x_one, x_zero


if __name__ == "__main__":

    # Define model.
    model_name = sys.argv[1]

    # Read in parameters for synthetic control method.
    sc_parameters = json.load(open(ppj(
            "IN_MODEL_SPECS",
            "synth_control_parameters_{}.json".format(model_name)),
            encoding="utf-8"))
    start_pre_treatment_time = sc_parameters["start_pre_treatment"]
    start_treatment_time = sc_parameters["start_treatment"]
    end_treatment_time = sc_parameters["end_treatment"]
    treated_unit = sc_parameters["treated_unit"]
    control_units = sc_parameters["donor_units"]
    name_dep = sc_parameters["name_dep_variable"]

    # Read in table of countries.
    filename = 'countries.csv'
    countries = pd.read_csv(ppj('OUT_DATA', filename), sep=',')
    countries = countries.set_index('country_id')

    # Read in table of periods.
    filename = 'periods.csv'
    periods = pd.read_csv(ppj('OUT_DATA', filename), sep=',')
    periods = periods.set_index('period_id')

    # Read in the data for the dependent variable.
    filename = sc_parameters["filename_dep_variable"]
    dependent_variable = pd.read_csv(ppj('OUT_DATA', filename), sep=',')
    dependent_variable = dependent_variable.set_index(['country_id',
                                                       'period_id'])

    # Read in data of predictor variables.
    filename = 'predictor_data.csv'
    predictor_data = pd.read_csv(ppj('OUT_DATA', filename), sep=',')
    predictor_data = predictor_data.set_index(['country_id'])

    # Standarize data of the dependent variable and predictors.
    dependent_variable_standarized = standarize_data(dependent_variable)
    predictor_data_standarized = predictor_data.copy()
    predictor_data_standarized['trade_openness'] = \
        standarize_data(predictor_data_standarized['trade_openness'])
    predictor_data_standarized['industry_share'] = \
        standarize_data(predictor_data_standarized['industry_share'])
    predictor_data_standarized['inflation_rate'] = \
        standarize_data(predictor_data_standarized['inflation_rate'])
    predictor_data_standarized['schooling'] = \
        standarize_data(predictor_data_standarized['schooling'])

    # Get index of treated unit and treatment times.
    index_treated = pd.Index(countries).get_loc((treated_unit,))
    index_treatment_start = pd.Index(periods).get_loc((start_treatment_time,))
    index_treatment_end = pd.Index(periods).get_loc((end_treatment_time,))
    index_pre_treatment_start = pd.Index(periods).get_loc((
            start_pre_treatment_time,))

    # Get list of index for control units.
    index_donors = []
    for country in control_units:
        index_donors.append(pd.Index(countries).get_loc((country,)))

    # Read in information to define predictors.
    predictor_specs = json.load(open(ppj(
            "IN_MODEL_SPECS", "predictors_{}.json".format(model_name)),
        encoding="utf-8"))
    predictors = predictor_specs["predictors"]

    # Get Z_tilde and X_tilde.
    z_tilde = get_z_tilde(dep_variable=dependent_variable_standarized,
                          index_treated=index_treated,
                          index_donors=index_donors,
                          index_pre_treatment_start=index_pre_treatment_start,
                          index_treatment_start=index_treatment_start)
    x_tilde = get_x_tilde(dep_variable=dependent_variable_standarized,
                          predictor_data=predictor_data_standarized,
                          predictors=predictors, index_treated=index_treated,
                          index_donors=index_donors)

    # Run synthetic control method to get optimal weights.
    status, w_optimal, v_optimal, ids_sunny_donors = \
        determine_synthetic_control_weights(x_tilde=x_tilde, z_tilde=z_tilde)

    # Prepare data for plots and tables from not standarazised data.
    z_one_pre_treatment = get_z_one_from_data(dep_variable=dependent_variable,
                                              index_treated=index_treated,
                                              index_pre_treatment_start=index_pre_treatment_start,
                                              index_treatment_start=index_treatment_start)
    z_one_post_treatment = get_z_one_from_data(dep_variable=dependent_variable,
                                               index_treated=index_treated,
                                               index_pre_treatment_start=index_treatment_start,
                                               index_treatment_start=index_treatment_end+1)
    z_zero_pre_treatment = get_z_zero_from_data(dep_variable=dependent_variable,
                                                index_donors=index_donors,
                                                index_pre_treatment_start=index_pre_treatment_start,
                                                index_treatment_start=index_treatment_start)
    z_zero_post_treatment = get_z_zero_from_data(dep_variable=dependent_variable,
                                                 index_donors=index_donors,
                                                 index_pre_treatment_start=index_treatment_start,
                                                 index_treatment_start=index_treatment_end+1)
    x_one_pre_treatment, x_zero_pre_treatment = get_x_one_and_x_zero_from_data(
            dep_variable=dependent_variable, predictor_data=predictor_data,
            predictors=predictors,
            index_treated=index_treated, index_donors=index_donors)

    # Caculate data for plots and tables for synthetic control unit.
    z_sc_pre_treatment = z_zero_pre_treatment @ w_optimal.T
    z_sc_post_treatment = z_zero_post_treatment @ w_optimal.T
    x_sc_pre_treatment = x_zero_pre_treatment @ w_optimal.T

    # Join pre-treatment and post-treatment data.
    z_one_whole_period = np.concatenate((z_one_pre_treatment,
                                         z_one_post_treatment), axis=0)
    z_sc_whole_period = np.concatenate((z_sc_pre_treatment,
                                        z_sc_post_treatment), axis=0)

    # Get the names for the predictors to store in solutions.
    names_predictors = []
    for dictionary in predictors:
        names_predictors.append(dictionary['name'])

    # Save data in a dictionary to store it in a pickle.
    solution_data = {'z_one': z_one_whole_period,
                     'z_sc': z_sc_whole_period,
                     'weights': w_optimal,
                     'x_one': x_one_pre_treatment,
                     'x_sc': x_sc_pre_treatment,
                     'start_date': start_pre_treatment_time,
                     'end_date': end_treatment_time,
                     'name_dep': name_dep,
                     'treatment_date': start_treatment_time,
                     'donor_countries': control_units,
                     'names_predictors': names_predictors,
                     'ids_sunny': ids_sunny_donors,
                     'v_opt': v_optimal,
                     'status': status}

    # Store data for plots and tables for each model.
    with open(ppj("OUT_ANALYSIS", "sc_{}.pickle".format(model_name)),
              "wb") as out_file:
        pickle.dump(solution_data, out_file)
        
    # Print status for each model to be able to check how the calculation is 
    # working during the compilation.
    print('Status of the model {} is:'.format(model_name), '\n', status)
