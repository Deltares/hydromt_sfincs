.. _model_methods:

Model Methods
=============

As mentioned before, each component of the SFINCS model has a ``read``, ``write`` and a
``create()`` method. The ``create()`` method is used to generate or update the component data
from external data sources, such as GIS data or other model outputs.
An overview of the most used SFINCS model methods is provided in the tables below.
When using HydroMT from the command line, only the create methods are exposed. Click on
a specific method to see its documentation.

General methods
---------------

.. _general_setup_table:

.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.read`
     - Read SFINCS model component from file.
   * - :py:func:`~hydromt_sfincs.SfincsModel.write`
     - Write SFINCS model component to file.
   * - :py:func:`~hydromt_sfincs.SfincsModel.build`
     - Build the complete SFINCS model by creating all components from yaml file.
   * - :py:func:`~hydromt_sfincs.SfincsModel.update`
     - Update SFINCS model component with new data from yaml file.
   * - :py:func:`~hydromt_sfincs.SfincsModel.plot_basemap`
     - Plot a basemap of the SFINCS model grid with optional components overlayed.
   * - :py:func:`~hydromt_sfincs.SfincsModel.plot_forcing`
     - Plot time series of the forcing data for the SFINCS model.

Config methods
--------------

.. _config_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:meth:`config.update <hydromt_sfincs.components.config.config.SfincsConfig.update>`
     - Update SFINCS config (sfincs.inp) with a dictionary.

Grid create methods
-------------------

.. _grid_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:meth:`grid.create <hydromt_sfincs.components.grid.regulargrid.SfincsGrid.create>`
     - This component generates a user-defined model grid.
   * - :py:meth:`grid.create_from_region <hydromt_sfincs.components.grid.regulargrid.SfincsGrid.create_from_region>`
     - This component automatically generates a model grid covering the region of interest with a given resolution.
   * - :py:meth:`elevation.create <hydromt_sfincs.components.grid.elevation.SfincsElevation.create>`
     - This component interpolates topobathy (depfile) data to the model grid.
   * - :py:meth:`mask.create_active <hydromt_sfincs.components.grid.mask.SfincsMask.create_active>`
     - This component generates a mask (mskfile) defining which part of the model grid is active based on elevation criteria and/or polygons.
   * - :py:meth:`mask.create_boundary <hydromt_sfincs.components.grid.mask.SfincsMask.create_boundary>`
     - This component adds boundary cells in the model mask (mskfile) based on elevation criteria and/or polygons.
   * - :py:meth:`roughness.create <hydromt_sfincs.components.grid.roughness.SfincsRoughness.create>`
     - This component adds a Manning roughness map (manningfile) to the model grid based on gridded Manning roughness data or a
       combinataion of gridded land-use/land-cover map and a Manning roughness mapping table.
   * - :py:meth:`infiltration.create_constant <hydromt_sfincs.components.grid.infiltration.SfincsInfiltration.create_constant>`
     - This component adds a spatially varying constant infiltration rate map (qinffile) to the model grid.
   * - :py:meth:`infiltration.create_cn <hydromt_sfincs.components.grid.infiltration.SfincsInfiltration.create_cn>`
     - This component adds a potential maximum soil moisture retention map (scsfile) to the model grid based on a gridded curve number map.
   * - :py:meth:`infiltration.create_cn_with_recovery <hydromt_sfincs.components.grid.infiltration.SfincsInfiltration.create_cn_with_recovery>`
     - This component adds a three layers related to the curve number (maximum and effective infiltration capacity; seff and smax) and
       saturated hydraulic conductivity (ks, to account for recovery) to the model
       grid based on landcover, Hydrological Similarity Group and saturated hydraulic conductivity (Ksat).
   * - :py:meth:`storage_volume.create <hydromt_sfincs.components.grid.storage_volume.SfincsStorageVolume.create>`
     - This component adds a storage volume map (volfile) to the model grid to account for green-infrastructure.
   * - :py:meth:`subgrid.create <hydromt_sfincs.components.grid.subgrid.SfincsSubgridTable.create>`
     - This component generates subgrid tables (sbgfile) for the model grid based on a list of elevation and Manning roughness datasets

Geometries create methods
-------------------------

.. _geoms_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`observation_points.create <hydromt_sfincs.SfincsModel.observation_points.create>`
     - This component adds observation points to the model (obsfile).
   * - :py:func:`cross_sections.create <hydromt_sfincs.SfincsModel.cross_sections.create>`
     - This component adds cross-sections to the model (crsfile).
   * - :py:func:`thin_dams.create <hydromt_sfincs.SfincsModel.thin_dams.create>`
     - This component adds line element structures to the model (thdfile).
   * - :py:func:`weirs.create <hydromt_sfincs.SfincsModel.weirs.create>`
     - This component adds line element structures to the model (weirfile).
   * - :py:func:`drainage_structures.create <hydromt_sfincs.SfincsModel.drainage_structures.create>`
     - This component adds drainage structures (pump, culvert, one-way-valve) to the model (drnfile).

Forcing create methods
----------------------

.. _forcing_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`water_level.create_boundary_points_from_mask <hydromt_sfincs.components.forcing.watet_level.SfincsWaterLevel.create_boundary_points_from_mask>`
     - This component adds waterlevel boundary points (bndfile) along model waterlevel boundary (msk=2).
   * - :py:func:`water_level.create <hydromt_sfincs.SfincsModel.setup_waterlevel_forcing>`
     - This component adds waterlevel forcing (bndfile, bzsfile) from a `geodataset` (geospatial point timeseries) or a tabular `timeseries` dataframe.
   * - :py:func:`~hydromt_sfincs.SfincsModel.rivers.create_inflow`
     - This component adds boundary cells in the model mask (mskfile) where a river flows out of the model domain.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_river_inflow`
     - This component adds discharge points (srcfile) where a river enters the model domain.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing`
     - This component adds discharge forcing (srcfile, disfile) from a `geodataset` (geospatial point timeseries) or a tabular `timeseries` dataframe.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing_from_grid`
     - This component adds discharge forcing (srcfile, disfile) based on a gridded discharge dataset.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_precip_forcing`
     - This component adds spatially uniform precipitation forcing from timeseries/constants (precipfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_precip_forcing_from_grid`
     - This component adds precipitation forcing from a gridded spatially varying data source (netamprfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_pressure_forcing_from_grid`
     - This component adds pressure forcing from a gridded spatially varying data source (netampfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_wind_forcing`
     - This component adds spatially uniform wind forcing from timeseries/constants (wndfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_wind_forcing_from_grid`
     - This component adds wind forcing from a gridded spatially varying data source (netamuamvfile).

Other create methods
--------------------

.. _other_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_tiles`
     - This component generates webmercator index and topobathy tiles for visualization of the SFINCS model.

.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
