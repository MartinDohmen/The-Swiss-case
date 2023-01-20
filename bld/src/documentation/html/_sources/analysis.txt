.. _analysis:

************************************
Main model estimations / simulations
************************************

In this section the main analysis is done. This means the code computes the synthetic control unit and all relevant values for every model.
To do so it defines numpy matrices from the data and calculate x_tilde and z_tilde which are needed as inputs for the algorithm used for the model code.
Then it calls the main function from the *model_code*, **determine_synthetic_control_weights**, to calculate the synthetic control. Afterwards, the data is formated as needed
for plots and graphs and all results are saved in pickle file as a dictionary.

All this is done in the file :file:`synthetic_control.py` in the folder *src.analysis*.

It does and contains:

.. automodule:: src.analysis.synthetic_control
    :members:
