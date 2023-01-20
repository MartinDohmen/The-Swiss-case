.. _data_management:

***************
Data management
***************

This section does include all code doing the data management. This means it cleans the data, replaces missing values and brings it in the second normal form.
Only the data needed for the later analysis is extracted.

The second normal form is violated in one case, namely in the created table containing the predictor variables. In this table also the country names are included.
This is done to make the data easier to read for humans in between, which was necessary in some stages of the project. As the dataset is only small, in my opinion the violation is not that much of a concern.

The data management is done in the file :file:`clean_raw_data.py` in the folder *src.data_management*.

It does and contains:

.. automodule:: src.data_management.clean_raw_data
    :members:
