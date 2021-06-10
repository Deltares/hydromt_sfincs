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

   sfincsModel

Build components
----------------

.. autosummary::
   :toctree: ../generated/

   sfincsModel.setup_config
   sfincsModel.setup_basemaps
   sfincsModel.setup_river_inflow
   sfincsModel.setup_river_outflow
   sfincsModel.setup_gauges
   sfincsModel.setup_structures
   sfincsModel.setup_manning_roughness
   sfincsModel.setup_cn_infiltration
   sfincsModel.setup_h_forcing
   sfincsModel.setup_q_forcing
   sfincsModel.setup_q_forcing_from_grid
   sfincsModel.setup_p_forcing
   sfincsModel.setup_p_forcing_from_grid

Plot methods
------------

.. autosummary::
   :toctree: ../generated/

   sfincsModel.plot_basemap
   sfincsModel.plot_forcing

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   sfincsModel.region
   sfincsModel.crs
   sfincsModel.res
   sfincsModel.root
   sfincsModel.config
   sfincsModel.staticmaps
   sfincsModel.staticgeoms
   sfincsModel.forcing
   sfincsModel.states
   sfincsModel.results

High level methods
------------------

.. autosummary::
   :toctree: ../generated/

   sfincsModel.read
   sfincsModel.write
   sfincsModel.build
   sfincsModel.update
   sfincsModel.set_root

Low level methods
-----------------

.. autosummary::
   :toctree: ../generated/

   sfincsModel.update_spatial_attrs
   sfincsModel.get_spatial_attrs

General methods
---------------

.. autosummary::
   :toctree: ../generated/

   sfincsModel.setup_config
   sfincsModel.get_config
   sfincsModel.set_config
   sfincsModel.read_config
   sfincsModel.write_config

   sfincsModel.set_staticmaps
   sfincsModel.read_staticmaps
   sfincsModel.write_staticmaps

   sfincsModel.set_staticgeoms
   sfincsModel.read_staticgeoms
   sfincsModel.write_staticgeoms

   sfincsModel.set_forcing
   sfincsModel.read_forcing
   sfincsModel.write_forcing

   sfincsModel.set_states
   sfincsModel.read_states
   sfincsModel.write_states

   sfincsModel.set_results
   sfincsModel.read_results


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

   utils.get_spatial_attrs
   utils.parse_datetime