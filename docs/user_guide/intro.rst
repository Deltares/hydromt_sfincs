.. _intro_user_guide:

User guide
==========

With the **HydroMT SFINCS plugin**, users can easily benefit from the rich set of tools of the 
`HydroMT package <https://github.com/Deltares/hydromt>`_ to build and update 
`SFINCS <https://sfincs.readthedocs.io/en/latest/>`_ models from available global and local data.

This plugin assists the SFINCS modeller in:

- Quickly setting up a base SFINCS model and default parameter values.
- Making maximum use of the best available global or local data.
- Adjusting and updating components of a SFINCS model and their associated parameters in a consistent way.
- Connecting SFINCS to other models (input from e.g. Wflow, output towards e.g. Delft-FIAT)
- Visualizing SFINCS models.
- Converting SFINCS schematizations to GIS formats.
- Analysing SFINCS model outputs.

Ways to use HydroMT-SFINCS
==========

There are 2 main ways to use HydroMT to build your SFINCS model:

- **1. Configuration file (basic user)**: 
      Provide some information in a text-file about the model to build, and build that model quickly in the command Line.

- **2. Python functions (advanced user)**: 
      Dive into the underlying Python functions, and use those to build your model from scratch in a Python script for the highest level of flexibility and automatisation.

The following explanation on available methods are therefore separated for both ways of making a SFINCS model.

.. toctree::
   :maxdepth: 2
   :hidden:

   sfincs.rst
   sfincs_methods.rst
   sfincs_build_update.rst
   run_model.rst
   process_analyze.rst