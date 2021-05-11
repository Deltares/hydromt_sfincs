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

   sfincs.SfincsModel

Build components
----------------

.. autosummary::
   :toctree: ../generated/

   sfincs.SfincsModel.setup_config
   sfincs.SfincsModel.setup_basemaps
   sfincs.SfincsModel.setup_rivers
   sfincs.SfincsModel.setup_gauges
   sfincs.SfincsModel.setup_manning_roughness
   sfincs.SfincsModel.setup_cn_infiltration
   sfincs.SfincsModel.setup_h_forcing
   sfincs.SfincsModel.setup_q_forcing
   sfincs.SfincsModel.setup_q_forcing_from_grid
   sfincs.SfincsModel.setup_p_forcing_gridded

Plot methods
------------

.. autosummary::
   :toctree: ../generated/

   sfincs.SfincsModel.plot_basemap
   sfincs.SfincsModel.plot_forcing

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   sfincs.SfincsModel.region
   sfincs.SfincsModel.crs
   sfincs.SfincsModel.res
   sfincs.SfincsModel.root
   sfincs.SfincsModel.config
   sfincs.SfincsModel.staticmaps
   sfincs.SfincsModel.staticgeoms
   sfincs.SfincsModel.forcing

High level methods
------------------

.. autosummary::
   :toctree: ../generated/

   sfincs.SfincsModel.read
   sfincs.SfincsModel.write
   sfincs.SfincsModel.build
   sfincs.SfincsModel.set_root

General methods
---------------

.. autosummary::
   :toctree: ../generated/

   sfincs.SfincsModel.setup_config
   sfincs.SfincsModel.get_config
   sfincs.SfincsModel.set_config
   sfincs.SfincsModel.read_config
   sfincs.SfincsModel.write_config

   sfincs.SfincsModel.set_staticmaps
   sfincs.SfincsModel.read_staticmaps
   sfincs.SfincsModel.write_staticmaps

   sfincs.SfincsModel.set_staticgeoms
   sfincs.SfincsModel.read_staticgeoms
   sfincs.SfincsModel.write_staticgeoms

   sfincs.SfincsModel.set_forcing
   sfincs.SfincsModel.read_forcing
   sfincs.SfincsModel.write_forcing
