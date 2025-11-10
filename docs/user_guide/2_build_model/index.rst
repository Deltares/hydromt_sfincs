.. _building_a_sfincs_model:

Building a SFINCS Model
=======================

When making a SFINCS model, you need to create multiple input files.
With the HydroMT SFINCS plugin, you can easily make these SFINCS model schematizations.
This plugin helps you preparing or updating several model components of a SFINCS model
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and
discharge forcing.

Each SFINCS model is represented by an instance of the :py:class:`~hydromt_sfincs.SfincsModel`,
which contains several components (see :ref:`model_components`). These components can be created
or updated using specific methods (see :ref:`model_methods`), which is demenstrated in the
:ref:`model_build` section.

.. toctree::
   :hidden:
   :maxdepth: 2

   model_components.rst
   model_methods.rst
   model_build.rst
