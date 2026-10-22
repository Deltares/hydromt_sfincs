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

The regular-grid component's ``data`` property contains the complete gridded
dataset. The individual grid components (for example, elevation, mask, and
roughness) also provide a ``data`` property, but that property exposes only the
layer or layers relevant to that component. These are component-specific views
of data stored on the shared grid; they are not separate data stores. To add or
replace a layer, use the parent grid's ``set()`` method.

The grid reads and writes the cell index, mask, and elevation maps. Other
component-owned maps are read and written by their components through the
grid's internal layer I/O methods. When ``SfincsGrid.write()`` is called with
``write_components=True``, it also calls the owning components' ``write()``
methods. Regular-grid infiltration layers are stored in their configured
binary map files; subgrid tables are managed by the subgrid component.

.. autosummary::
   :toctree: ../_generated/

   components.grid.SfincsGrid
   components.grid.SfincsGrid.data
   components.grid.SfincsGrid.read
   components.grid.SfincsGrid.write
   components.grid.SfincsGrid.create
   components.grid.SfincsGrid.create_from_region
   components.grid.SfincsGrid._read_layers
   components.grid.SfincsGrid._write_layers
   components.grid.SfincsGrid.read_map
   components.grid.SfincsGrid.write_map

   components.grid.SfincsElevation
   components.grid.SfincsElevation.data
   components.grid.SfincsElevation.create

   components.grid.SfincsMask
   components.grid.SfincsMask.data
   components.grid.SfincsMask.create_active
   components.grid.SfincsMask.create_boundary

   components.grid.SfincsRoughness
   components.grid.SfincsRoughness.data
   components.grid.SfincsRoughness.read
   components.grid.SfincsRoughness.write
   components.grid.SfincsRoughness.create

   components.grid.SfincsInfiltration
   components.grid.SfincsInfiltration.data
   components.grid.SfincsInfiltration.read
   components.grid.SfincsInfiltration.write
   components.grid.SfincsInfiltration.create_uniform_constant
   components.grid.SfincsInfiltration.create_constant
   components.grid.SfincsInfiltration.create_cn
   components.grid.SfincsInfiltration.create_cn_from_landuse_hsg
   components.grid.SfincsInfiltration.create_cn_with_recovery
   components.grid.SfincsInfiltration.create_green_ampt
   components.grid.SfincsInfiltration.create_green_ampt_from_maps
   components.grid.SfincsInfiltration.create_horton
   components.grid.SfincsInfiltration.create_horton_from_maps

   components.grid.SfincsInitialConditions
   components.grid.SfincsInitialConditions.data
   components.grid.SfincsInitialConditions.read
   components.grid.SfincsInitialConditions.write
   components.grid.SfincsInitialConditions.create

   components.grid.SfincsStorageVolume
   components.grid.SfincsStorageVolume.data
   components.grid.SfincsStorageVolume.read
   components.grid.SfincsStorageVolume.write
   components.grid.SfincsStorageVolume.create

   components.grid.SfincsSubgridTable
   components.grid.SfincsSubgridTable.data
   components.grid.SfincsSubgridTable.read
   components.grid.SfincsSubgridTable.write
   components.grid.SfincsSubgridTable.create

Quadtree
--------

The quadtree grid's ``data`` property contains the complete mesh dataset.
Individual quadtree components expose their relevant layer or layers through
their own ``data`` properties. These are component-specific views of data
stored on the shared quadtree grid, not separate data stores; use the parent
grid's ``set()`` method to add or replace a layer. The component views retain
the mesh topology.

The mask, elevation, and any quadtree-specific variables are stored in the main
quadtree file (``qtrfile``). Other component-owned variables are written to
standalone UGRID netCDF files by their components. Calling
``SfincsQuadtreeGrid.write()`` with ``write_components=True`` also invokes those
component writers. Quadtree infiltration, including bucket variables, is
stored in ``inffile``; ``inftype`` identifies the flavor. The subgrid table is
managed separately by its subgrid component.

.. autosummary::
   :toctree: ../_generated/

   components.quadtree.SfincsQuadtreeGrid
   components.quadtree.SfincsQuadtreeGrid.data
   components.quadtree.SfincsQuadtreeGrid.read
   components.quadtree.SfincsQuadtreeGrid._read_layers
   components.quadtree.SfincsQuadtreeGrid._write_layers
   components.quadtree.SfincsQuadtreeGrid.write
   components.quadtree.SfincsQuadtreeGrid.create
   components.quadtree.SfincsQuadtreeGrid.create_from_region

   components.quadtree.SfincsQuadtreeElevation
   components.quadtree.SfincsQuadtreeElevation.data
   components.quadtree.SfincsQuadtreeElevation.read
   components.quadtree.SfincsQuadtreeElevation.write
   components.quadtree.SfincsQuadtreeElevation.create
   components.quadtree.SfincsQuadtreeElevation.create_uniform

   components.quadtree.SfincsQuadtreeMask
   components.quadtree.SfincsQuadtreeMask.data
   components.quadtree.SfincsQuadtreeMask.read
   components.quadtree.SfincsQuadtreeMask.write
   components.quadtree.SfincsQuadtreeMask.create_active
   components.quadtree.SfincsQuadtreeMask.create_boundary

   components.quadtree.SfincsQuadtreeRoughness
   components.quadtree.SfincsQuadtreeRoughness.data
   components.quadtree.SfincsQuadtreeRoughness.read
   components.quadtree.SfincsQuadtreeRoughness.write
   components.quadtree.SfincsQuadtreeRoughness.create

   components.quadtree.SfincsQuadtreeInfiltration
   components.quadtree.SfincsQuadtreeInfiltration.data
   components.quadtree.SfincsQuadtreeInfiltration.read
   components.quadtree.SfincsQuadtreeInfiltration.write
   components.quadtree.SfincsQuadtreeInfiltration.create_uniform_constant
   components.quadtree.SfincsQuadtreeInfiltration.create_constant
   components.quadtree.SfincsQuadtreeInfiltration.create_cn
   components.quadtree.SfincsQuadtreeInfiltration.create_cn_from_landuse_hsg
   components.quadtree.SfincsQuadtreeInfiltration.create_cn_with_recovery
   components.quadtree.SfincsQuadtreeInfiltration.create_green_ampt
   components.quadtree.SfincsQuadtreeInfiltration.create_green_ampt_from_maps
   components.quadtree.SfincsQuadtreeInfiltration.create_horton
   components.quadtree.SfincsQuadtreeInfiltration.create_horton_from_maps
   components.quadtree.SfincsQuadtreeInfiltration.create_bucket
   components.quadtree.SfincsQuadtreeInfiltration.create_bucket_from_maps
   components.quadtree.SfincsQuadtreeInfiltration.create_runoff_coefficient

   components.quadtree.SfincsQuadtreeInitialConditions
   components.quadtree.SfincsQuadtreeInitialConditions.data
   components.quadtree.SfincsQuadtreeInitialConditions.read
   components.quadtree.SfincsQuadtreeInitialConditions.write
   components.quadtree.SfincsQuadtreeInitialConditions.create
   components.quadtree.SfincsQuadtreeInitialConditions.create_from_polygon

   components.quadtree.SfincsQuadtreeStorageVolume
   components.quadtree.SfincsQuadtreeStorageVolume.data
   components.quadtree.SfincsQuadtreeStorageVolume.read
   components.quadtree.SfincsQuadtreeStorageVolume.write
   components.quadtree.SfincsQuadtreeStorageVolume.create

   components.quadtree.SfincsQuadtreeSubgridTable
   components.quadtree.SfincsQuadtreeSubgridTable.read
   components.quadtree.SfincsQuadtreeSubgridTable.write
   components.quadtree.SfincsQuadtreeSubgridTable.create

   components.quadtree.SnapWaveQuadtreeMask
   components.quadtree.SnapWaveQuadtreeMask.data
   components.quadtree.SnapWaveQuadtreeMask.create
   components.quadtree.SnapWaveQuadtreeMask.create_active
   components.quadtree.SnapWaveQuadtreeMask.create_boundary

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
   workflows.create_topobathy_tiles

Infiltration
------------

Estimation maths only; these take already-read arrays and tables and return
SFINCS parameter maps. Reading data and writing config is done by the
infiltration components.

.. autosummary::
   :toctree: ../_generated/

   workflows.ksat_to_mmhr
   workflows.normalize_hsg_codes
   workflows.cn_to_s
   workflows.curve_number_from_landuse_hsg
   workflows.adjust_curve_number
   workflows.curve_number_with_recovery
   workflows.constant_infiltration_from_ksat_lulc
   workflows.green_ampt_from_hsg
   workflows.green_ampt_from_hsg_landuse
   workflows.horton_from_hsg
   workflows.horton_from_hsg_landuse
   workflows.bucket_from_hsg
   workflows.bucket_from_hsg_landuse

Flood map downscaling
---------------------

.. autosummary::
   :toctree: ../_generated/

   workflows.downscaling.downscale_floodmap
   workflows.downscaling.adjust_zsmax_dilation
   workflows.downscaling.adjust_zsmax_energyhead
   workflows.downscaling.remove_disconnected_flooding
   workflows.downscaling.make_index_cog
   workflows.downscale_floodmap_webmercator

.. _methods:

SFINCS low-level methods
========================

Input/Output methods
---------------------

.. autosummary::
   :toctree: ../_generated/

   readers.read_binary_map
   writers.write_binary_map
   readers.read_binary_map_index
   writers.write_binary_map_index
   readers.read_ascii_map
   writers.write_ascii_map
   readers.read_timeseries
   writers.write_timeseries
   readers.read_xy
   writers.write_xy
   readers.read_xyn
   writers.write_xyn
   readers.read_geoms
   writers.write_geoms
   readers.read_drn
   writers.write_drn
   readers.read_sfincs_map_results
   readers.read_sfincs_his_results

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
