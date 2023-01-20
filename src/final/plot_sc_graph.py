"""
Plot the graph describing the solution of the synthetic control method.
This means plot a line graph with time on x-achsis and the dependent variable
on y-achsis and lines for the treated unit and the synthetic control.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pickle
import sys
import numpy as np

from bld.project_paths import project_paths_join as ppj


def get_dates_for_x_axsis(start_year, end_year, start_quarter=1,
                          end_quarter=1):
    """Prepare the dates on the x-axis of the plot."""

    x_dates = []
    for year in range(start_year, end_year+1):
        if year == start_year:
            for quarter in range(start_quarter, 5):
                date = datetime.date(year, quarter * 3 - 2, 1)
                x_dates.append(date)
        elif year == end_year:
            for quarter in range(1, end_quarter+1):
                date = datetime.date(year, quarter * 3 - 2, 1)
                x_dates.append(date)
        else:
            for quarter in range(1, 5):
                date = datetime.date(year, quarter * 3 - 2, 1)
                x_dates.append(date)

    return x_dates


def plot_dep_var(z_one, z_sc, model_name, start_date, end_date, name,
                 treatment_date):
    """Create a line graph of the dependent variable for the treated unit and
    the synthetic control unit with time on x-axis.
    """

    # Define the basic characteristics of the figure.
    fig, ax = plt.subplots()

    # Create labels and interval for x-axis.
    years = mdates.YearLocator()
    months = mdates.MonthLocator(interval=3)
    yearsFmt = mdates.DateFormatter('%Y')

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    datemin = datetime.date(int(start_date[3:7]), 3*int(start_date[1])-2, 1)
    datemax = datetime.date(2017, 12, 31)
    if model_name == 'gdppc_time_placebo':
        datemax = datetime.date(2007, 12, 31)
    ax.set_xlim(datemin, datemax)

    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

    # Rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them.
    fig.autofmt_xdate()

    # Define the dates for the x-axis to match with the data.
    x = get_dates_for_x_axsis(int(start_date[3:7]), int(end_date[3:7]),
                              int(start_date[1]), int(end_date[1]))

    # Reshape data as 1d-array for plotting.
    z_one_for_plot = np.reshape(z_one, z_one.size)
    z_sc_for_plot = np.reshape(z_sc, z_sc.size)

    # Set bounds or dependent variable.
    if name == "GDP per Capita in const. prices":
        if model_name == 'gdppc_country_placebo':
            ax.set_ylim(30000, 50000)
        else:
            ax.set_ylim(40000, 60000)

    # Plot the data and labels.
    ax.plot(x, z_one_for_plot, 'k-', label='treated country')
    ax.plot(x, z_sc_for_plot, 'b--', label='synthetic control')
    plt.xlabel('Year')
    plt.ylabel(name)
    ax.legend(loc='lower right')

    treatment_date = datetime.date(int(treatment_date[3:7]),
                                   3*int(treatment_date[1])-2, 1)
    plt.axvline(x=treatment_date, color='r', linestyle=':', linewidth=0.7)

    # Save the figure.
    plt.savefig(ppj("OUT_FIGURES", "sc_graph_{}.pdf".format(model_name)))


if __name__ == "__main__":
    model_name = sys.argv[1]

    # Load data to plot.
    with open(ppj("OUT_ANALYSIS", "sc_{}.pickle".format(model_name)),
              "rb") as in_file:
        solution_data = pickle.load(in_file)

    plot_dep_var(solution_data['z_one'], solution_data['z_sc'], model_name,
                 solution_data['start_date'], solution_data['end_date'],
                 solution_data['name_dep'], solution_data['treatment_date'])
