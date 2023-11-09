.. currentmodule:: hydromt_sfincs


.. _api_reference:

=============
API reference
=============

.. _api_model:

SFINCS model class
==================

Initialize
----------

.. autosummary::
   :toctree: _generated/

   SfincsModel

.. _components:

Setup components
----------------

.. autosummary::
   :toctree: _generated/

   SfincsModel.setup_config
   SfincsModel.setup_region
   SfincsModel.setup_grid
   SfincsModel.setup_grid_from_region
   SfincsModel.setup_dep
   SfincsModel.setup_mask_active
   SfincsModel.setup_mask_bounds
   SfincsModel.setup_manning_roughness
   SfincsModel.setup_constant_infiltration
   SfincsModel.setup_cn_infiltration
   SfincsModel.setup_cn_infiltration_with_kr
   SfincsModel.setup_subgrid
   SfincsModel.setup_river_inflow
   SfincsModel.setup_river_outflow
   SfincsModel.setup_observation_points
   SfincsModel.setup_observation_lines
   SfincsModel.setup_structures
   SfincsModel.setup_drainage_structures
   SfincsModel.setup_storage_volume
   SfincsModel.setup_waterlevel_forcing
   SfincsModel.setup_waterlevel_bnd_from_mask
   SfincsModel.setup_discharge_forcing
   SfincsModel.setup_discharge_forcing_from_grid
   SfincsModel.setup_precip_forcing
   SfincsModel.setup_precip_forcing_from_grid
   SfincsModel.setup_pressure_forcing_from_grid
   SfincsModel.setup_wind_forcing
   SfincsModel.setup_wind_forcing_from_grid
   SfincsModel.setup_tiles

Plot methods
------------

.. autosummary::
   :toctree: _generated/

   SfincsModel.plot_basemap
   SfincsModel.plot_forcing

Attributes
----------

.. autosummary::
   :toctree: _generated/

   SfincsModel.region
   SfincsModel.mask
   SfincsModel.crs
   SfincsModel.res
   SfincsModel.root
   SfincsModel.config
   SfincsModel.grid
   SfincsModel.geoms
   SfincsModel.forcing
   SfincsModel.states
   SfincsModel.results

High level methods
------------------

.. autosummary::
   :toctree: _generated/

   SfincsModel.read
   SfincsModel.write
   SfincsModel.build
   SfincsModel.update
   SfincsModel.set_root

Low level methods
-----------------

.. autosummary::
   :toctree: _generated/

   SfincsModel.update_grid_from_config
   SfincsModel.update_spatial_attrs
   SfincsModel.set_forcing_1d
   SfincsModel.get_model_time

General methods
---------------

.. autosummary::
   :toctree: _generated/

   SfincsModel.setup_config
   SfincsModel.get_config
   SfincsModel.set_config
   SfincsModel.read_config
   SfincsModel.write_config

   SfincsModel.set_grid
   SfincsModel.read_grid
   SfincsModel.write_grid

   SfincsModel.read_subgrid
   SfincsModel.write_subgrid

   SfincsModel.set_geoms
   SfincsModel.read_geoms
   SfincsModel.write_geoms

   SfincsModel.set_forcing
   SfincsModel.read_forcing
   SfincsModel.write_forcing

   SfincsModel.set_states
   SfincsModel.read_states
   SfincsModel.write_states

   SfincsModel.set_results
   SfincsModel.read_results

.. _workflows:

SFINCS workflows
================

.. autosummary::
   :toctree: _generated/

   workflows.merge_multi_dataarrays
   workflows.merge_dataarrays
   workflows.burn_river_rect
   workflows.snap_discharge
   workflows.river_boundary_points
   workflows.river_centerline_from_hydrography
   workflows.landuse
   workflows.cn_to_s
   workflows.create_topobathy_tiles
   workflows.downscale_floodmap_webmercator

.. _methods:

SFINCS low-level methods
========================

Input/Output methods
---------------------

.. autosummary::
   :toctree: _generated/

   utils.read_binary_map
   utils.write_binary_map
   utils.read_binary_map_index
   utils.write_binary_map_index
   utils.read_ascii_map
   utils.write_ascii_map
   utils.read_timeseries
   utils.write_timeseries
   utils.read_xy
   utils.write_xy
   utils.read_xyn
   utils.write_xyn
   utils.read_geoms
   utils.write_geoms
   utils.read_drn
   utils.write_drn
   utils.read_sfincs_map_results
   utils.read_sfincs_his_results

Utilities
---------

.. autosummary::
   :toctree: _generated/

   utils.parse_datetime
   utils.gdf2linestring
   utils.linestring2gdf
   utils.gdf2polygon
   utils.polygon2gdf
   utils.get_bounds_vector
   utils.mask2gdf
   utils.rotated_grid 

Visualization
-------------

.. autosummary::
   :toctree: _generated/

   plots.plot_basemap
   plots.plot_forcing
   utils.downscale_floodmap
   workflows.downscale_floodmap_webmercator