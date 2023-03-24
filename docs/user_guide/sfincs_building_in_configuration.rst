.. currentmodule:: hydromt_sfincs.sfincs

.. _model_methods_configuration:

============================
Methods using configuration file
============================

With the HydroMT SFINCS plugin, you can easily work with SFINCS model schematizations. 
This plugin helps you preparing or updating several model components of a SFINCS model 
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and 
discharge forcing.

.. _model_methods:

Model setup methods
===================

The SFINCS model setup methods are available from the configuration file and 
allow you to configure HydroMT in order to build or update SFINCS model schematizations.

A short example of how these methods can be called in the configuration file is showed below.

.. literalinclude:: ../_examples/sfincs_simple_compound_base.ini
   :language: Ini

So in this example the called setup methods were:

- setup_config

- setup_grid_from_region

- setup_dep

- setup_mask_active

- setup_mask_bounds

An overview of all the available SfincsModel setup methods is provided in the table below. 

.. Tip::
  For more elaborate examples of how to use this configuration file see: :ref:`examples`

Overview of model setup methods
===================

Click on header to get a full overview or directly on a specific method see its documentation.  

.. _setup_table:

.. list-table:: Setup methods
   :widths: 20 25
   :header-rows: 1

   * - SFINCS file
     - :ref:`model setup methods <model_methods>`
   * - model region
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_region`
   * - sfincs.inp
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_config`
   * - depfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_dep` :py:func:`~hydromt_sfincs.SfincsModel.setup_river_bathymetry`
   * - mskfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_active` :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_bounds` :py:func:`~hydromt_sfincs.SfincsModel.setup_river_outflow`
   * - sbgfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_subgrid`
   * - manningfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_manning_roughness`
   * - scsfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_cn_infiltration`
   * - obsfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_observation_points`
   * - thd- & weirfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_structures`

.. _forcing_setup_table:

.. list-table:: Forcing setup methods
   :widths: 20 25
   :header-rows: 1

   * - SFINCS file
     - :ref:`model setup methods <model_methods>`
   * - bnd- & bzsfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_waterlevel_forcing`
   * - src- & disfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing`
   * - precipfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_p_forcing`
   * - netamprfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_p_forcing_from_grid`

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html