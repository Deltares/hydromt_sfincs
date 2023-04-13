=============================
Working with the SFINCS model
=============================

There are 2 main ways to use HydroMT to build your SFINCS model:

**1. Command Line Interface (basic user)**: 
      Provide some information about the model configuration in a .yml-file and quickly build the model using the Command Line Interface (CLI).

**2. Python scripting (advanced user)**: 
      Dive into the underlying Python functions and use those to build your model from scratch in a Python script. 
      This option is recommended when the user wants to (locally) adjust the model input data before building the model, 
      e.g. in the case of modifications of the bed levels or variations on the boundary conditions.

.. toctree::
    :maxdepth: 2
    :hidden:

    sfincs_model_setup.rst
    sfincs_build_update.rst
    sfincs_run.rst

