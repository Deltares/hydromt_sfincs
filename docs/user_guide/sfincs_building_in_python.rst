.. currentmodule:: hydromt_sfincs.sfincs

.. _model_methods_python:

Methods using Python scripts
===================

The SFINCS model methods are available as Python functions and 
allow you to configure HydroMT in order to build or update SFINCS model schematizations.

An overview of the available SfincsModel methods, workflows and low-level methods
is provided in the table below. Click on header to get a full overview or directly on
a specific method see its documentation.  

A short example of how these methods can be called in Python is showed below.

.. code-block:: console

    from hydromt_sfincs import SfincsModel
  
    sf = SfincsModel(data_libs=["artifact_data"], root="sfincs_compound")

    sf.create_grid(grid_type="regular", **inp_dict)

    sf.plot_basemap()

    sf.write() # write all

.. Tip::
  For more elaborate examples of how to use this configuration file see: :ref:`examples`

Overview of model setup methods
===================

.. _general_table:

.. list-table:: General setup methods
   :widths: 20 20 20 20 25
   :header-rows: 1

   * - SFINCS file
     - :ref:`model setup methods <model_methods>`
     - :ref:`model create methods <model_create>`
     - :ref:`workflows <workflows>`
     - :ref:`low-level methods <methods>`
   * - model region
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_region`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_grid`
     - :py:func:`~hydromt.workflows.parse_region` :py:func:`~hydromt.workflows.get_basin_geometry`
     - 
   * - sfincs.inp
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_config`
     - :py:func:`~hydromt_sfincs.SfincsModel.set_config`
     - :py:func:`~hydromt.workflows.parse_region`:sup:`1` :py:func:`~hydromt.workflows.get_basin_geometry`:sup:`1`
     - :py:func:`~hydromt_sfincs.read_inp` :py:func:`~hydromt_sfincs.write_inp` :py:func:`~hydromt_sfincs.get_spatial_attrs`
   * - depfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_dep` :py:func:`~hydromt_sfincs.SfincsModel.setup_river_bathymetry`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_dep`
     - :py:func:`~hydromt_sfincs.workflows.merge_topobathy`
     - :py:func:`~hydromt_sfincs.read_binary_map` :py:func:`~hydromt_sfincs.write_binary_map`
   * - mskfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_active` :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_bounds` :py:func:`~hydromt_sfincs.SfincsModel.setup_river_outflow`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_mask_active` :py:func:`~hydromt_sfincs.SfincsModel.create_mask_bounds`
     - :py:func:`~hydromt_sfincs.utils.mask_topobathy` :py:func:`~hydromt_sfincs.utils.mask_bounds`
     - :py:func:`~hydromt_sfincs.read_binary_map` :py:func:`~hydromt_sfincs.write_binary_map` 
   * - indfile
     - 
     - 
     - 
     - :py:func:`~hydromt_sfincs.read_binary_map_index` :py:func:`~hydromt_sfincs.write_binary_map_index`
   * - sbgfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_subgrid`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_subgrid`
     - 
     - 
   * - manningfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_manning_roughness`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_manning_roughness`
     - :py:func:`~hydromt_sfincs.workflows.landuse`
     - :py:func:`~hydromt_sfincs.read_binary_map` :py:func:`~hydromt_sfincs.write_binary_map`
   * - scsfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_cn_infiltration`
     - 
     - :py:func:`~hydromt_sfincs.workflows.cn_to_s`
     - :py:func:`~hydromt_sfincs.read_binary_map` :py:func:`~hydromt_sfincs.write_binary_map`
   * - obsfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_observation_points`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_observation_points`
     -
     - :py:func:`~hydromt_sfincs.read_xy` :py:func:`~hydromt_sfincs.write_xy`
   * - thd- & weirfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_structures`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_structures`
     -
     - :py:func:`~hydromt_sfincs.read_structures` :py:func:`~hydromt_sfincs.write_structures` :py:func:`~hydromt_sfincs.utils.gdf2structures` :py:func:`~hydromt_sfincs.utils.structures2gdf`

.. _forcing_table:

.. list-table:: Forcing setup methods
   :widths: 20 20 20 20 25
   :header-rows: 1

   * - SFINCS file
     - :ref:`model setup methods <model_methods>`
     - :ref:`model create methods <model_create>`
     - :ref:`workflows <workflows>`
     - :ref:`low-level methods <methods>`
   * - bnd- & bzsfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_waterlevel_forcing`
     - :py:func:`~hydromt_sfincs.SfincsModel.create_waterlevel_forcing`
     -
     - :py:func:`~hydromt_sfincs.read_timeseries` :py:func:`~hydromt_sfincs.write_timeseries` :py:func:`~hydromt_sfincs.read_xy` :py:func:`~hydromt_sfincs.write_xy`
   * - src- & disfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_river_inflow` :py:func:`~hydromt_sfincs.SfincsModel.setup_q_forcing` :py:func:`~hydromt_sfincs.SfincsModel.setup_q_forcing_from_grid`
     - :py:func:`~hydromt_sfincs.workflows.create_discharge_forcing`
     - :py:func:`~hydromt_sfincs.workflows.snap_discharge`
     - :py:func:`~hydromt_sfincs.read_timeseries` :py:func:`~hydromt_sfincs.write_timeseries` :py:func:`~hydromt_sfincs.read_xy` :py:func:`~hydromt_sfincs.write_xy`
   * - precipfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_p_forcing` :py:func:`~hydromt_sfincs.SfincsModel.setup_p_forcing_from_grid`
     - 
     - :py:func:`~hydromt.workflows.resample_time`:sup:`1`
     - :py:func:`~hydromt_sfincs.read_timeseries` :py:func:`~hydromt_sfincs.write_timeseries`
   * - netamprfile
     - :py:func:`~hydromt_sfincs.SfincsModel.setup_p_forcing_from_grid`
     - 
     - :py:func:`~hydromt.workflows.resample_time`:sup:`1`
     -

:sup:`1`) Imported from hydromt core package

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html