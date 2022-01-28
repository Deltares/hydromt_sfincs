.. currentmodule:: hydromt_sfincs

=============
API reference
=============

.. _api_model:

SFINCS model class
==================

Initialize
----------

.. autosummary::
   :toctree: ../generated/

   SfincsModel

.. _components:

Setup components
----------------

.. autosummary::
   :toctree: ../generated/

   SfincsModel.setup_config
   SfincsModel.setup_topobathy
   SfincsModel.setup_merge_topobathy
   SfincsModel.setup_mask
   SfincsModel.setup_bounds
   SfincsModel.setup_river_hydrography
   SfincsModel.setup_river_bathymetry
   SfincsModel.setup_river_inflow
   SfincsModel.setup_river_outflow
   SfincsModel.setup_manning_roughness
   SfincsModel.setup_cn_infiltration
   SfincsModel.setup_gauges
   SfincsModel.setup_structures
   SfincsModel.setup_h_forcing
   SfincsModel.setup_q_forcing
   SfincsModel.setup_q_forcing_from_grid
   SfincsModel.setup_p_forcing
   SfincsModel.setup_p_forcing_from_grid

Plot methods
------------

.. autosummary::
   :toctree: ../generated/

   SfincsModel.plot_basemap
   SfincsModel.plot_forcing

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   SfincsModel.region
   SfincsModel.mask
   SfincsModel.crs
   SfincsModel.res
   SfincsModel.root
   SfincsModel.config
   SfincsModel.staticmaps
   SfincsModel.staticgeoms
   SfincsModel.forcing
   SfincsModel.states
   SfincsModel.results

High level methods
------------------

.. autosummary::
   :toctree: ../generated/

   SfincsModel.read
   SfincsModel.write
   SfincsModel.build
   SfincsModel.update
   SfincsModel.set_root

Low level methods
-----------------

.. autosummary::
   :toctree: ../generated/

   SfincsModel.update_spatial_attrs
   SfincsModel.get_spatial_attrs
   SfincsModel.set_forcing_1d
   SfincsModel.get_model_time

General methods
---------------

.. autosummary::
   :toctree: ../generated/

   SfincsModel.setup_config
   SfincsModel.get_config
   SfincsModel.set_config
   SfincsModel.read_config
   SfincsModel.write_config

   SfincsModel.set_staticmaps
   SfincsModel.read_staticmaps
   SfincsModel.write_staticmaps

   SfincsModel.set_staticgeoms
   SfincsModel.read_staticgeoms
   SfincsModel.write_staticgeoms

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
   :toctree: ../generated/

   workflows.merge_topobathy
   workflows.get_rivbank_dz
   workflows.get_river_bathymetry
   workflows.burn_river_zb
   workflows.snap_discharge
   workflows.river_boundary_points
   workflows.landuse
   workflows.cn_to_s

.. _methods:

SFINCS low-level methods
========================

Input/Output methods
---------------------

.. autosummary::
   :toctree: ../generated/

   read_inp
   write_inp
   read_binary_map
   write_binary_map
   read_binary_map_index
   write_binary_map_index
   read_ascii_map
   write_ascii_map
   read_timeseries
   write_timeseries
   read_xy
   write_xy
   read_structures
   write_structures
   read_sfincs_map_results
   read_sfincs_his_results

Utilities
---------

.. autosummary::
   :toctree: ../generated/

   utils.mask_topobathy
   utils.mask_bounds
   utils.get_spatial_attrs
   utils.parse_datetime
   utils.gdf2structures
   utils.structures2gdf

Visualization
-------------

.. autosummary::
   :toctree: ../generated/

   plots.plot_basemap
   plots.plot_forcing