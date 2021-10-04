What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

Bugfix
^^^^^^
- bugfix setup_p_forcing

Changed
^^^^^^^
- ``setup_mask``: removed default values and new option to combine a polygon and elevation based criteria.

v0.2.0 (2 August 2021)
---------------------

Bugfix
^^^^^^
- scsfile variable changed to maximum soil moisture retention [inch]; was curve number [-]
- fix setting delimited text based geodatasets for h and q forcing.

Changed
^^^^^^^
- Bumped minimal hydromt vesion to 0.4.2
- splitted ``setup_basemaps`` into multiple smaller methods: ``setup_merge_topobathy``, ``setup_mask`` and ``setup_bounds``
- separated many low-level methods into utils.py and plots.py
- save bzs/bzd & dis/src only as GeoDataArray at forcing and do not copy the locations at staticgeoms.
- sort src/bnd files on x_dim for comparibility between OS
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