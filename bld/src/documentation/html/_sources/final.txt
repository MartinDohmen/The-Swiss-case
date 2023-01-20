.. _final:

************************************
Visualisation and results formatting
************************************


In this section the tables and figures are constructed from the results contained in the pickle file saved in the main analysis.
The code is located in the directory *src.final*.

Tables
==========

For every specification the file :file:`sc_tables.py` construct three tables:

1. a table containing the weights for the countries in the donor pool constituting the synthetic control.
2. a table containing the predictors for the treated country and the synthetic control.
3. a table containing the outcomes for the treated country and the synthetic control as well as the constraint weighting matrix V and the status of the optimization. (This table is constructed just for inspection and to calculate some values for the paper, but is not included in the paper)


Figures
==========

Furthermore, for every specification the file :file:`plot_sc_graph.py` construct a figure showing the development of the outcome variable for the treated unit and the synthetic control.
To do so it defines and uses to functions.

The file does and contains:

.. automodule:: src.final.plot_sc_graph
    :members:
