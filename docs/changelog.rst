What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

v0.2.1 (23 February 2022)
-------------------------

Deprecated
^^^^^^^^^^
- **setup_basemaps** has been replaced by **setup_topobathy**
- In **setup_mask**, the "active_mask_fn" argument has been renamed to "mask_fn" for consistency
- In **setup_river_inflow** and **setup_river_outflow** the "basemaps_fn" argument has been renamed to "hydrography_fn" for consistency
- In **setup_river_outflow** the "outflow_width" argument has been renamed to "river_width" for consistency with setup_river_inflow
- **setup_q_forcing_from_grid** and **workflows.snap_discharge** have a "rel_error" and "abs_error" argument instead of a single "max_error" argument.

Bugfix
^^^^^^
- bugfix **setup_p_forcing** to ensure the data is 1D when passed to set_forcing_1d method
- bugfix **setup_p_forcing_from_grid** when aggregating with a multi polygon region.
- bugfix **read_results** with new corner_x/y instead of edge_x/y dimensions in sfincs_map.nc

New
^^^
- **setup_region** method to set the (hydrological) model region of interest (before part of **setup_basemaps**).
- **setup_river_hydrography** allows to derive hydrography data ['flwdir', 'uparea'] from the model elevation or reproject it from a global dataset.
  Derived 'uparea' and 'flwdir' maps are saved in the GIS folder and can be reused later (if kept together with the model)
- **setup_river_bathymetry** to estimate a river depth based on bankfull discharge and river width. A mask of river cells 'rivmsk' is kept in the GIS folder.

Changed
^^^^^^^
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
----------------------

Bugfix
^^^^^^
- scsfile variable changed to maximum soil moisture retention [inch]; was curve number [-]
- fix setting delimited text based geodatasets for h and q forcing.

Changed
^^^^^^^
- Bumped minimal hydromt version to 0.4.2
- splitted ``setup_topobathy`` into multiple smaller methods: ``setup_merge_topobathy``, ``setup_mask`` and ``setup_bounds``
- separated many low-level methods into utils.py and plots.py
- save bzs/bzd & dis/src only as GeoDataArray at forcing and do not copy the locations at staticgeoms.
- sort src/bnd files on x_dim for comparability between OS
- staticmaps are by default saved (and read) in S->N orientation as this matches the SFINCS better.


Added
^^^^^
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
^^^^^^^^^^^^^
- build from python example
- overviews with SfincsModel setup components & SfincsModel data

Deprecated
^^^^^^^^^^^
- ``setup_p_gridded``

v0.1.0 (18 May 2021)
--------------------
Noticeable changes are a new ``setup_river_inflow`` and ``setup_river_outflow`` methods

Added
^^^^^

- setup_river_outflow method to set ouflow (msk=3) boundary at river outflow points

Changed
^^^^^^^

- Updated to hydromt v0.4.1


Documentation
^^^^^^^^^^^^^

- Now **latest** and **stable** versions.
- Updated build instructions
- Added **build_coastal_model**, **build_riverine_model** and **plot_sfincs_map** notebooks to the examples.


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html