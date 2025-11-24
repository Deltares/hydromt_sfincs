.. currentmodule:: hydromt_sfincs

.. _api_reference:

===
API
===

.. _api_model:

SFINCS Model class
==================

The ``hydromt_sfincs.SfincsModel`` class is the main entry point to read, write, build, and update SFINCS models using HydroMT.
It uses the functionalities provided by the reusable components defined in the ``hydromt_sfincs.components`` module.

.. autosummary::
   :toctree: ../_generated/

   SfincsModel

Methods
-------

.. autosummary::
   :toctree: ../_generated/

   SfincsModel.read
   SfincsModel.write
   SfincsModel.build
   SfincsModel.update
   .. SfincsModel.set_root

Plot methods
------------

.. autosummary::
   :toctree: ../_generated/

   SfincsModel.plot_basemap
   SfincsModel.plot_forcing

Attributes
----------

.. autosummary::
   :toctree: ../_generated/

   SfincsModel.root
   SfincsModel.crs
   SfincsModel.region
   SfincsModel.bbox

.. _api_components:

Components
==========

The ``hydromt_sfincs.components`` module defines reusable data container classes
that represent configuration, grid, geometries, boundary conditions, outputs,
and other model data.

Configuration
-------------

.. autosummary::
   :toctree: ../_generated/

   components.config.SfincsConfig
   components.config.SfincsConfig.data
   components.config.SfincsConfig.read
   components.config.SfincsConfig.write
   components.config.SfincsConfig.update
   components.config.SfincsConfig.update_grid_from_config
   components.config.SfincsConfig.get
   components.config.SfincsConfig.set
   components.config.SfincsConfig.get_set_file_variable

   components.config.SfincsConfigVariables

Grid
----

.. autosummary::
   :toctree: ../_generated/

   components.grid.SfincsGrid
   components.grid.SfincsGrid.data
   components.grid.SfincsGrid.read
   components.grid.SfincsGrid.write
   components.grid.SfincsGrid.create
   components.grid.SfincsGrid.create_from_region

   components.grid.SfincsElevation
   components.grid.SfincsElevation.create

   components.grid.SfincsMask
   components.grid.SfincsMask.create_active
   components.grid.SfincsMask.create_boundary

   components.grid.SfincsRoughness
   components.grid.SfincsRoughness.create

   components.grid.SfincsInfiltration
   components.grid.SfincsInfiltration.create_constant
   components.grid.SfincsInfiltration.create_cn
   components.grid.SfincsInfiltration.create_cn_with_recovery

   components.grid.SfincsInitialConditions
   components.grid.SfincsInitialConditions.create

   components.grid.SfincsStorageVolume
   components.grid.SfincsStorageVolume.create

   components.grid.SfincsSubgridTable
   components.grid.SfincsSubgridTable.data
   components.grid.SfincsSubgridTable.read
   components.grid.SfincsSubgridTable.write
   components.grid.SfincsSubgridTable.create

Geometries
-----------

.. autosummary::
   :toctree: ../_generated/

   components.geometries.SfincsObservationPoints
   components.geometries.SfincsObservationPoints.data
   components.geometries.SfincsObservationPoints.read
   components.geometries.SfincsObservationPoints.write
   components.geometries.SfincsObservationPoints.create

   components.geometries.SfincsCrossSections
   components.geometries.SfincsCrossSections.data
   components.geometries.SfincsCrossSections.read
   components.geometries.SfincsCrossSections.write
   components.geometries.SfincsCrossSections.create

   components.geometries.SfincsThinDams
   components.geometries.SfincsThinDams.data
   components.geometries.SfincsThinDams.read
   components.geometries.SfincsThinDams.write
   components.geometries.SfincsThinDams.create

   components.geometries.SfincsWeirs
   components.geometries.SfincsWeirs.data
   components.geometries.SfincsWeirs.read
   components.geometries.SfincsWeirs.write
   components.geometries.SfincsWeirs.create

   components.geometries.SfincsDrainageStructures
   components.geometries.SfincsDrainageStructures.data
   components.geometries.SfincsDrainageStructures.read
   components.geometries.SfincsDrainageStructures.write
   components.geometries.SfincsDrainageStructures.create

Forcing
--------

.. autosummary::
   :toctree: ../_generated/

   components.forcing.SfincsWaterLevel
   components.forcing.SfincsWaterLevel.data
   components.forcing.SfincsWaterLevel.read
   components.forcing.SfincsWaterLevel.write
   components.forcing.SfincsWaterLevel.create
   components.forcing.SfincsWaterLevel.create_timeseries
   components.forcing.SfincsWaterLevel.create_timeseries_from_astro
   components.forcing.SfincsWaterLevel.create_boundary_points_from_mask

   components.forcing.SfincsDischargePoints
   components.forcing.SfincsDischargePoints.data
   components.forcing.SfincsDischargePoints.read
   components.forcing.SfincsDischargePoints.write
   components.forcing.SfincsDischargePoints.create
   components.forcing.SfincsDischargePoints.create_timeseries

   components.forcing.SfincsPrecipitation
   components.forcing.SfincsPrecipitation.data
   components.forcing.SfincsPrecipitation.read
   components.forcing.SfincsPrecipitation.write
   components.forcing.SfincsPrecipitation.create
   components.forcing.SfincsPrecipitation.create_uniform

   components.forcing.SfincsPressure
   components.forcing.SfincsPressure.data
   components.forcing.SfincsPressure.read
   components.forcing.SfincsPressure.write
   components.forcing.SfincsPressure.create

   components.forcing.SfincsWind
   components.forcing.SfincsWind.data
   components.forcing.SfincsWind.read
   components.forcing.SfincsWind.write
   components.forcing.SfincsWind.create
   components.forcing.SfincsWind.create_uniform

   components.forcing.SfincsRivers
   components.forcing.SfincsRivers.data
   components.forcing.SfincsRivers.read
   components.forcing.SfincsRivers.write
   components.forcing.SfincsRivers.create_river_inflow

Output
------

.. autosummary::
   :toctree: ../_generated/

   components.output.SfincsOutput

.. _workflows:

SFINCS workflows
================

.. autosummary::
   :toctree: ../_generated/

   workflows.merge_multi_dataarrays
   workflows.merge_dataarrays
   workflows.burn_river_rect
   workflows.snap_discharge
   workflows.river_source_points
   workflows.river_centerline_from_hydrography
   workflows.landuse
   workflows.cn_to_s
   workflows.create_topobathy_tiles

.. _methods:

SFINCS low-level methods
========================

Input/Output methods
---------------------

.. autosummary::
   :toctree: ../_generated/

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
   :toctree: ../_generated/

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
   :toctree: ../_generated/

   plots.plot_basemap
   plots.plot_forcing
   utils.downscale_floodmap
   workflows.downscale_floodmap_webmercator
