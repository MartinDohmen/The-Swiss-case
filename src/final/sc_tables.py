"""
Construct latex tables for each model to store and display:
        1. weights of the synthetic control unit
        2. predictors for treated and synthetic control.
"""

import pickle
import sys
import numpy as np

from bld.project_paths import project_paths_join as ppj


if __name__ == "__main__":
    model_name = sys.argv[1]

    # Load data to put in tables.
    with open(ppj("OUT_ANALYSIS", "sc_{}.pickle".format(model_name)),
              "rb") as in_file:
        solution_data = pickle.load(in_file)

    # Define necessary parameters from solutions for easier readability.
    weights = np.reshape(solution_data['weights'],
                         solution_data['weights'].size)
    donor_countries = solution_data['donor_countries']
    predictors_treated = np.reshape(solution_data['x_one'],
                                    solution_data['x_one'].size)
    predictors_sc = np.reshape(solution_data['x_sc'],
                               solution_data['x_sc'].size)
    predictor_names = solution_data['names_predictors']
    ids_sunny = solution_data['ids_sunny']
    dep_var_treated = solution_data['z_one']
    dep_var_sc = solution_data['z_sc']
    v_opt = solution_data['v_opt']
    status = solution_data['status']

    # Write the a LaTeX table containing the weights for the synthetic control
    # unit of that model.
    with open(ppj("OUT_TABLES", "weights_{}.tex".format(model_name)),
              'w') as tex_file:

        # Write command to begin table and row of headers with line at below.
        tex_file.write(r'\begin{tabular}{c|c|c}')
        tex_file.write('\n')
        tex_file.write(
                r'\textbf{Country}&\textbf{Sunny Donor}&\textbf{Weight}\\')
        tex_file.write('\n')
        tex_file.write(r'\hline ')
        tex_file.write('\n')

        # Write country and according weight in table plus indicator if country
        # is a sunny donor.
        for i in range(0, weights.size):
            if i in ids_sunny:
                sunny = 'Yes'
            else:
                sunny = 'No'
            tex_file.write('{} & {} & {:.4f} \\\\ \n'.format(
                donor_countries[i], sunny, weights[i]))

        # Write end of table.
        tex_file.write(r'\end{tabular}')

    # Write the LaTeX table containing the predictors for the treated and the
    # synthetic control unit.
    with open(ppj("OUT_TABLES", "predictors_{}.tex".format(model_name)),
              'w') as tex_file:

        # Write command to begin table and row of headers with line at below.
        tex_file.write(r'\begin{tabular}{c|c|c}')
        tex_file.write('\n')
        tex_file.write(
            r'predictor&\textbf{treated country}&\textbf{synthetic control}\\')
        tex_file.write('\n')
        tex_file.write(r'\hline ')
        tex_file.write('\n')

        # Write country and according weight in table.
        for i in range(0, predictors_treated.size):
            tex_file.write('{} & {:.2f} & {:.2f}\\\\\n'.format(
                predictor_names[i], predictors_treated[i], predictors_sc[i]))

        # Write end of table.
        tex_file.write(r'\hline')
        tex_file.write('\n')
        tex_file.write(r'\end{tabular}')

    # Write values of dependent variable as a test and to include in text.
    with open(ppj("OUT_TABLES", "dep_var_{}.tex".format(model_name)),
              'w') as tex_file:
        tex_file.write('dependent variable treated unit')
        tex_file.write(np.array_str(dep_var_treated))
        tex_file.write('\n\ndependent variable synthetic control')
        tex_file.write(np.array_str(dep_var_sc))
        tex_file.write('\n\n V:\n')
        tex_file.write(np.array_str(v_opt))
        tex_file.write('\n\n Status:\n')
        tex_file.write(status)
