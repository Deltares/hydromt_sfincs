.. _working_with_sfincs:

=============================
Working with the SFINCS model
=============================

There are 2 main ways to use HydroMT to build your SFINCS model:

**1. Command Line Interface (basic user)**:
      Provide some information about the model configuration in a .yml-file and quickly build the model using the Command Line Interface (CLI).
      The .yml-file provides a way to create a reproducible model setup recipe, which can be easily be shared with others.
      Additionally, no Python knowledge is required to use the CLI.

**2. Python scripting (advanced user)**:
      Dive into the underlying Python functions and use those to build your model from scratch in a Python script.
      This option is recommended for the expert user who wants to (locally) adjust the model input data as part of the model building process,
      e.g. in the case of in-memory modifications of the bed levels or variations on the boundary conditions.
      The Python interface provides a lot of flexibility and access to the full HydroMT-SFINCS API, but requires some knowledge of Python.

.. toctree::
    :maxdepth: 2
    :hidden:

    sfincs_model_setup.rst
    sfincs_build_update.rst
    sfincs_run.rst
