.. _updating_a_sfincs_model:

Updating a SFINCS Model
=======================

In the previous :ref:`building_a_sfincs_model` section, it was shown how to create a new SFINCS model from scratch.
It's good modelling practice to start with a basic model setup and gradually improve the model by adding or updating
specific components. This could be for example updating the infiltration component to include a more realistic infiltration approach,
or adding observation points to monitor water levels at specific locations in the model, or adding additional forcing data.


Similar to building a SFINCS model, there are two main ways to update an existing SFINCS model using HydroMT:
- the :ref:`command line interface <sfincs_cli>` for basic users, and
- :ref:`Python scripting <sfincs_python>` for advanced users.

In the following sections, examples are provided how to update your SFINCS model with HydroMT.

.. toctree::
   :maxdepth: 2

   model_update.rst
