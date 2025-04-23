What's new
==========
All notable changes to this project will be documented in this page.
Distinction is made between new methods (Added), changes to existing methods (Changed), bugfixes (Fixed), deprecated methods (Deprecated) and removed methods (Removed).

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

v1.2.0 (23-4-2025)
==================
**This release contains new functionality on reading/writing quadtree models and some minor bugfixes and updates.
It is strongly recommended to use this version in combination with
`SFINCS versions ≥ 2.2.0 <https://github.com/Deltares/SFINCS/releases/tag/v2.2.0_col_dEze_release>`.**


Added
-----
- added reading/writing of quadtree models, but NOT building them from scratch (#226)
- added a "factor_ksat" to `SfincsModel.setup_cn_infiltration_with_ks` to change the units of the saturated hydraulic conductivity (#227)
- added the option to provide cumulative precipitation [mm] instead of precipitation rates [mm/hr] to `SfincsModel.setup_precip_forcing_from_grid` (#255)

Changed
-------
- nr_subgrid_pixels in `SfincsModel.setup_subgrid` can only be a multiple of 2 (#225)
- provide decent warning when all waterlevel boundary points are outside of region+buffer (#237)
- default value for "tspinup" changed to 0.0 in sfincs.inp to align with SFINCS kernel (#243)
- no automatic upscaling anymore of meteo datasets with intervals smaller than 1 hour for wind, pressure and precipitation (#255)
- changed the way `SfincsModel.setup_precip_forcing_from_grid` deals with cumulative precipitation inputs (#255)

Fixed
-----
- fixed a bug in `river_centerline_from_hydrography` in case there is only one river with a single segment (#215)
- clean the grid variables when recreating the grid (#220)
- minor bugfix in downscale floodmap routine when providing a file_name (#224)
- fixed erronuous mapping of uv points to joined uv-array when writing the subgrid files (#225)
- unpinned numpy versions <2.0 in package requirements (#244)
- fixed missing projection method while merging datasets (#247)
- fixed burning in rivers when very few riverbed points are provided (#250)

v1.1.0 (05-09-2024)
===================
**This release contains some breaking changes and can only be used in combination with
`SFINCS versions ≥ 2.1.1  <https://github.com/Deltares/SFINCS/releases/tag/v2.1.1_Dollerup_release>`_.**

The most important change is the implementation of a new subgrid methodology including wet fraction as in Van Ormondt et al. (2024, in review),
which is now written as a NetCDF file! The old implementation is still available when providing the original binary file, but then all wet fractions are assumed to be 1.

Next to this, some minor additions and bugfixes are made that improve the overall functionality of the package.

Added
-----
- improved subgrid tables that account for the wet fraction of the cell (#160)
- add source points at headwater locations with `SfincsModel.setup_river_inflow` (#170)
- it's now possible to provide a format/precision in `SfincsModel.write_forcing` (#197)
- river bed levels for burn-in can now be provided at point locations in `SfincsModel.setup_subgrid` (#209)

Changed
-------
- improved subgrid tables are saved as NetCDF (#160)
- improved weighting of adjacent cells in u/v points for determining representative Manning roughness and conveyance depth (#200)
- In `SfincsModel.setup_river_inflow`, in case of a confluence within a user-defined buffer of the
  model boundary, the confluence rather than both tributaries is selected as inflow point. (#202)
- turned on "baro", the atmospheric pressure term in the momentum equation, in sfincs.inp by default (#208)
- the expected variable names for wind and pressure forcing have been changed to "wind10_u", "wind10_v" and "press_msl" to match hydromt-core conventions (#211)

Fixed
-----
- rounding errors in `workflows.tile_window` which resulted in erronuous slippy-tiles (#178)
- "active geometry column to use has not been set" error for GeoDataFrame (#180)
- added nodata value to raster before writing (#199)


v1.0.3 (3 January 2024)
=======================
This release contains several changes and fixes. Most notably, models with a geographical CRS can now be build.
Also note the changes in the default values in the sfincs.inp file.

Changed
-------
- support for HydroMT core v0.9
- changed default value of dtmaxout to 86400 (was 99999.0) and for advection to 1 (instead of non-existing 2) in sfincs.inp (#156)
- support for filtering forcing data from timeseries and locations based on model region (#162)

Fixed
-----
- fixed `SfincsModel.setup_subgrid` for models with geograpgical CRS (#152)
- fixed masking of elevation and manning datasets when providing mask attribute (#153)
- fix a bug that caused some model files to be read twice (#161)
- bugfix in workflows.bathymetry that was encountered while burning in rivers (#166)


v1.0.2 (9 November 2023)
========================
This release mostly contains bugfixes and improvements to existing methods. Some new features have been added as well,
such as the option to add storage volume to the model to account for green infrastructure. Another addition is the
generation of overviews for the Cloud Optimized GeoTIFF files for faster visualization.

Added
-----
- `SfincsModel.setup_storage_volume` to account for green-infrastructure PR #101
- Added reverse_river_geom keyword argument in workflows.river_boundary_points #PR 136
- the COG files that are written automatically contain overviews for faster visualization PR #144

Changed
-------
- Changed `setup_cn_infiltration_with_kr` into `setup_cn_infiltration_with_ks` since saturated hydraulic conductivity (ks) is used instead of recovery rate (kr) PR #126
- precision of coordinates in geoms and forcing now depends on CRS (geographic or not) PR #143


Fixed
-----
- writing COG files in `SfincsModel.setup_subgrid` (the COG driver settings were wrong) PR #117
- a constant offset in the `datasets_dep` argument to `SfincsModel.setup_subgrid` and `SfincsModel.setup_dep` was ignored PR #119
- mismatch between gis data and the model grid causing issues while reading the model PR #128
- Bugfixes in workflows.river_boundary_points to make sure function also works with geoDataFrame #PR 136
- `utils.downscale_floodmap` now also works for large (rotated) grids PR #145

Deprecated
----------


v1.0.1 (3 August 2023)
======================
This release contains several new features, such as burning in river bathymetry into the subgrid, setting up drainage structures and adding wind and pressure forcing.
It also contains several bugfixes and improvements to existing methods.
It is recommended to use this release together with the latest version of the `SFINCS model <https://github.com/Deltares/SFINCS/releases/tag/v2.0.2_Blockhaus_release>`_.

Added
-----
- `SfincsModel.setup_cn_infiltration_with_kr` to setup three layers related to the curve number
  (maximum and effective infiltration capacity; seff and smax) and recovery rate (kr) PR #87
- `SfincsModelsetup_drainage_structures` to setup drainage structures (pumps,culverts) from a geodataframe. PR#90
- Added `SfincsModel.setup_wind_forcing`, `SfincsModel.setup_wind_forcing_from_grid` and `SfincsModel.setup_pressure_forcing_from_grid` methods to easily add wind and pressure forcing.  PR #92
- `SfincsModel.setup_observation_lines` to setup model observation lines (cross-sections) to monitor discharges. PR #114

Changed
-------
- `SfincsModel.setup_subgrid` now supports the 'riv_datasets' to burn in river bathymetry into the subgrid. PR #84
- `SfincsModel.setup_mask_active` argument reset_mask default to True PR #94
- `SfincsModel.read_config` allows to use a template input file from a directory different than the model root. PR #102
- Added the option to use landuse/landcover data combined with a reclass table to `SfincsModel.setup_constant_infiltration`.  PR #103
- Enabled to provide locations only (so no timeseries) for `SfincsModel.setup_waterlevel_forcing` and `SfincsModel.setup_discharge_forcing` PR #104
- New optional buffer argument in  `SfincsModel.setup_discharge_forcing` to select gauges around boundary only. PR #104
- `SfincsModel.plot_basemaps` now supports other CRS than UTM zones. PR #111
- New functionality within `SfincsModel.setup_structures` to use high resolution dem for weir elevation. PR #109
- hydromt_data.yml is written to the model root directory with used data sources.

Fixed
------
- bugfix in `SfincsModel.write_forcing` to ensure all NetCDF files are written instead of only the first one. PR #86
- bugfix in `SfincsModel.read_config` & `SfincsInput.read` for relative paths in inp file. PR #88
- bugfix in `SfincsModel.setup_subgrid` to ensure that a 'big geotiff' will be written by default when 'write_dep_tif' or 'write_man_tif' are True
- fix memory issues caused by rasterizing the model region and reprojecting before clipping of rasters. PR #94
- bugfix in `Sfincs.read_forcing` when combining attributes from the locations stored in the gis folder with the actual forcing locations. PR #99
- bugfix in `SfincsModel.setup_discharge_from_grid` when snapping based on upstream area in case a src points is outside of the uparea grid domain. PR #99

Removed
----------
- `burn_river_zb` and `get_river_bathymetry` workflow methods have been deprecated in favor of `burn_river_rect`. PR #84

v1.0 (17 April 2023)
====================

This release is a major update of the SfincsModel interface. It contains many new features,
such as support for *rotated grids*, *subgrid* and improved support for *building models from Python* scripts.
The documentation and exmaples have been updated to reflect these changes.

The release however also contains several breaking changes as we have tried to improve the
consistency of the interface and match it more closely to the SFINCS model itself.
Please carefully check the API reference for the new methods and arguments.

Main differences
----------------
- `setup_region` has been replaced by `setup_grid_from_region` and  `setup_grid`.
  This method actually creates an empty regular grid based on a region of interest or user-defined coordinates, shape, rotation, etc..
- `setup_dep` has replaced `setup_topobathy` and `setup_merge_topobathy`.
  This method can now also be used to setup a bathymetry map from multiple sources at once.
- `setup_mask_active` has replaced `setup_mask`.
- `setup_mask_bounds` has replaced `setup_bounds`
- `setup_waterlevel_forcing` has replaced `setup_h_forcing` and now supports merging fording from several data sources
- `setup_discharge_forcing` has replaced `setup_q_forcing` and now supports merging fording from several data sources
- `setup_discharge_forcing_from_grid` has replaces `setup_q_forcing_from_grid`
- `setup_precip_forcing` has replaced `setup_p_forcing`
- `setup_precip_forcing_from_grid` has replaced `setup_p_forcing_from_grid`
- `setup_observation_points` has replace `setup_gauges`

Added
-----------
- `setup_grid` to setup a user-defined regular grid based coordinates, shape, rotation, etc.
- `setup_subgrid` to setup subgrid tables (sbgfile) based on one ore more elevation and Manning roughness datasets
- `setup_constant_infiltration` to setup a constant infiltration rate maps (qinffile)
- `setup_waterlevel_bnd_from_mask` to setup water level boundary points (bndfile) based on the SFINCS model mask (mskfile)
- `setup_tiles` to create tiles of the model for fast visualization

Changed
---------------
- `setup_river_inflow` and `setup_river_outflow` are now based river centerline data (which can be derivded from hydrography data).
  This is more robust compared to the previous method which was based on reprojected flow direction data.

Removed (not replaced)
------------------------------
- `setup_basemaps` This method was already deprecated in v0.2.1 and has now been removed.
- `setup_river_hydrography` This method was removed as reprojection of the hydrography data is no longer required for river inflow/outflow.
- `setup_river_bathymetry` This method was removed as river bathymetry should ideally be burned in the subgrid data of the model rather
  than the dep file itself to be able to include rivers with widths smaller than the model grid cell size. A new option to burn rivers
  in the subgrid data will be added in to `setup_subgrid` a future release.


New low-level classes
---------------------
These classes are not intended to be used directly by the user, but are used internally by the SfincsModel class.

- The `SfincsInput` class contains methods to generate, read and write SFINCS input files
- The `RegularGrid` class contains methods to create and manipulate regular grids
- The `SubgridTableRegular` class contains methods to create and manipulate subgrid tables for regular grids


v0.2.1 (23 February 2022)
=========================

Deprecated
----------
- **setup_basemaps** has been replaced by **setup_topobathy**
- In **setup_mask**, the "active_mask_fn" argument has been renamed to "mask_fn" for consistency
- In **setup_river_inflow** and **setup_river_outflow** the "basemaps_fn" argument has been renamed to "hydrography_fn" for consistency
- In **setup_river_outflow** the "outflow_width" argument has been renamed to "river_width" for consistency with setup_river_inflow
- **setup_q_forcing_from_grid** and **workflows.snap_discharge** have a "rel_error" and "abs_error" argument instead of a single "max_error" argument.

Bugfix
------
- bugfix **setup_p_forcing** to ensure the data is 1D when passed to set_forcing_1d method
- bugfix **setup_p_forcing_from_grid** when aggregating with a multi polygon region.
- bugfix **read_results** with new corner_x/y instead of edge_x/y dimensions in sfincs_map.nc

New
---
- **setup_region** method to set the (hydrological) model region of interest (before part of **setup_basemaps**).
- **setup_river_hydrography** allows to derive hydrography data ['flwdir', 'uparea'] from the model elevation or reproject it from a global dataset.
  Derived 'uparea' and 'flwdir' maps are saved in the GIS folder and can be reused later (if kept together with the model)
- **setup_river_bathymetry** to estimate a river depth based on bankfull discharge and river width. A mask of river cells 'rivmsk' is kept in the GIS folder.
- Added parameter mapping file for ESA Worldcover dataset

Changed
-------
- **setup_mask** and **setup_bounds** both have a "mask_fn", "include_mask_fn" and "exclude_mask_fn" polygon and "min_elv" and "max_elv" elevation arguments to determine valid / boundary cells.
- **setup_mask** and **setup_bounds** have a "reset_mask" and "reset_bounds" option respectively to start with a clean mask or remove previously set boundary cells.
- **setup_mask** takes a new "drop_area" argument to drop regions of contiguous cells smaller than this maximum area threshold, useful to remove (spurious) small islands.
- **setup_mask** takes a new "fill_area" argument to fill regions of contiguous cells below the "min_elv" or above "max_elv" threshold surrounded by cells within the valid elevation range.
- In **setup_bounds** and **setup_mask** a "connectivity" argument is exposed to determine whether edge cells or regions of contiguous cells should be based on D4 (horizontal and vertical) or D8 (also diagonal) connections.
- In **setup_bounds** we avoid open boundary cells (mask == 3) next to water level boundary cells (mask == 2)
- **setup_merge_topobathy** has a new "max_width" argument to use bathymetry data from new source within a fixed width around the topography data.
- **setup_river_inflow** and **setup_river_outflow** are now based on the same **workflows.river_boundary_points** method.
   Both have a "river_upa" and "river_len" argument and the hydrography data is not required if **setup_river_hydrography** is ran beforehand.
   The model domain is also determined on-the-fly, thus it is not required to run setup_mask beforehand.
- **setup_river_inflow** has a new "river_width" argument to ensure closed boundary cells near a discharge source location
- **write_config** has a new "rel_path" argument that allows you to write sfincs.inp with references to model files in the root and rel_path directory.
- Write dep file with cm accuracy. This should be sufficient but also hides differences between linux and window builds.
- Exposed "interp_method" argument in **setup_merge_topobathy** to select interpolation method to fill NaNs.
- **setup_cn_infiltration** and **setup_manning_roughness** use default values for river cells as defined in **setup_river_bathymetry**
- The **setup_manning_rougness** has a new "sea_man" argument to set a constant roughness for cells below zero elevation.
- An improved version of interbasins **region** option has been implemented, see hydroMT core v0.4.5 for details.
- Bumped minimal pyflwdir version to 0.5.5
- Use mamba to setup CI environments


v0.2.0 (2 August 2021)
======================

Bugfix
------
- scsfile variable changed to maximum soil moisture retention [inch]; was curve number [-]
- fix setting delimited text based geodatasets for h and q forcing.

Changed
-------
- Bumped minimal hydromt version to 0.4.2
- splitted ``setup_topobathy`` into multiple smaller methods: ``setup_merge_topobathy``, ``setup_mask`` and ``setup_bounds``
- separated many low-level methods into utils.py and plots.py
- save bzs/bzd & dis/src only as GeoDataArray at forcing and do not copy the locations at staticgeoms.
- sort src/bnd files on x_dim for comparability between OS
- staticmaps are by default saved (and read) in S->N orientation as this matches the SFINCS better.


Added
-----
support for SFINCS files:

- structures: sfincs.thd & sfincs.weir
- results: sfincs_map.nc & sfincs_his.nc
- states: sfincs.restart
- forcing: sfincs.precip

new methods:

- ``setup_p_forcing_from_grid`` and ``setup_p_forcing`` with support for spatial uniform precip
- ``setup_merge_topobathy`` to merge a new topo/bathymetry dataset with the basemap DEM
- ``setup_mask`` and ``setup_bounds`` methods to setup the sfincs mask file
- ``setup_structures`` thd/weir files are read/written as part of read_staticgeoms
- ``read_states``, ``write_states`` methods with support for restart
- ``read_results``
- ``update_spatial_attrs`` and ``get_spatial_attrs`` (previously part of read_staticmaps)

new workflows:

- ``merge_topobathy``
- ``mask_topobathy``
- ``snap_discharge``
- ``river_inflow_points`` & ``river_outflow_points``

Documentation
-------------
- build from python example
- overviews with SfincsModel setup components & SfincsModel data

Deprecated
-----------
- ``setup_p_gridded``

v0.1.0 (18 May 2021)
====================
Noticeable changes are a new ``setup_river_inflow`` and ``setup_river_outflow`` methods

Added
-----

- setup_river_outflow method to set ouflow (msk=3) boundary at river outflow points

Changed
-------

- Updated to hydromt v0.4.1


Documentation
-------------

- Now **latest** and **stable** versions.
- Updated build instructions
- Added **build_coastal_model**, **build_riverine_model** and **plot_sfincs_map** notebooks to the examples.


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
