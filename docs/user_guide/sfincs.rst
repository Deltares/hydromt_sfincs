.. _model_components:
.. currentmodule:: hydromt_sfincs.sfincs

=======================
SFINCS model components
=======================

With the hydromt_sfincs plugin, you can easily work with SFINCS model schematizations. 
This plugin helps you preparing or updating  several model components of a SFINCS model 
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and 
discharge forcing.

When building or updating a model from command line a model region_; a model setup 
configuration (.ini file) with model components_ and options and, optionally, 
a data_ sources (.yml) file should be prepared.

The SFINCS model components are available from the HydroMT Command Line and Python Interfaces and 
allow you to configure HydroMT in order to build or update SFINCS model schematizations.
See :ref:`Coastal SFINCS model schematization <sfincs_coastal>` and 
:ref:`Riverine SFINCS model schematization <sfincs_riverine>` for suggested components
and options to use for coastal or riverine applications.

For python users all SFINCS attributes and methods are available, see :ref:`api_model`

.. _components:

Model components
================

The following components are available to build or update SFINCS model schematizations:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   SfincsModel.setup_config
   SfincsModel.setup_basemaps
   SfincsModel.setup_river_inflow
   SfincsModel.setup_river_outflow
   SfincsModel.setup_gauges
   SfincsModel.setup_manning_roughness
   SfincsModel.setup_cn_infiltration
   SfincsModel.setup_h_forcing
   SfincsModel.setup_q_forcing
   SfincsModel.setup_q_forcing_from_grid
   SfincsModel.setup_p_forcing_gridded


.. warning::

    In SFINCS, the order in which the components are listed in the ini file is important: 
    `setup_river_inflow` should be run before `setup_q_forcing` or `setup_q_forcing_from_grid`.

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options
