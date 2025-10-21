============================
Model components and methods
============================

When making a SFINCS model, you need to create multiple input files.
With the HydroMT SFINCS plugin, you can easily make these SFINCS model schematizations.
This plugin helps you preparing or updating several model components of a SFINCS model
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and
discharge forcing.

.. _model_components:


Model Components
================

The :py:class:`~hydromt_sfincs.SfincsModel` consists of several components that together
represent the full SFINCS model setup. Each component manages a specific part of the
model (e.g., configuration, grid definition, forcings, or output) and can be read from or
written to disk using the corresponding ``read()`` and ``write()`` methods.

For more details about each component, see the `SFINCS documentation <https://sfincs.readthedocs.io/en/latest/>`_.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - **Component**
     - **Description**
     - **Associated Files / Relations**
   * - :py:class:`~hydromt_sfincs.components.config.config.SfincsConfig`
     - Model configuration settings and parameters.
     - ``sfincs.inp`` (main input file)
   * - :py:class:`~hydromt_sfincs.components.output.SfincsOutput`
     - Model simulation results
     - ``sfincs_his.nc``, ``sfincs_map.nc``

**Regular grid components**

Define the model grid and its physical parameters. These are interrelated (e.g., ``grid`` and ``elevation`` share spatial resolution).

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - **Component**
     - **Description**
     - **Associated Files / Relations**
   * - :py:attr:`~hydromt_sfincs.SfincsModel.grid`
     - Base model grid.
     - ``sfincs.inp``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.elevation`
     - Elevation (bathymetry/topography).
     - ``depfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.mask`
     - Domain and boundary mask.
     - ``mskfile``, ``indexfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.roughness`
     - Manning’s n roughness values.
     - ``manningfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.infiltration`
     - Infiltration capacity map.
     - ``qinffile``, ``scsfile``, ``ksfile``, ``sefffile``, ``smaxfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.storage_volume`
     - Storage and volume correction.
     - ``volfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.initial_conditions`
     - Model initial states.
     - ``inifile`` (optional)
   * - :py:attr:`~hydromt_sfincs.SfincsModel.subgrid`
     - Subgrid table with cell-specific elevation and flow data.
     - ``sbgfile``


**Geometries and structures**
Vector datasets defining observation points, structures, and flow features.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - **Component**
     - **Description**
     - **Associated Files / Relations**
   * - :py:attr:`~hydromt_sfincs.SfincsModel.observation_points`
     - Observation points for validation.
     - ``obsfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.cross_sections`
     - Cross-section definitions.
     - ``crsfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.weirs`
     - Weir locations and parameters.
     - ``weirfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.thin_dams`
     - Thin dams and barriers.
     - ``thdfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.drainage_structures`
     - Drainage infrastructure.
     - ``drnfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.rivers`
     - River network and flow attributes.
     - Not used by the SFINCS model directly

**Forcing components**
Time-varying boundary and meteorological forcings applied to the model.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - **Component**
     - **Description**
     - **Associated Files / Relations**
   * - :py:attr:`~hydromt_sfincs.SfincsModel.water_level`
     - Water level boundary conditions.
     - ``bndfile``, ``bzsfile``, ``netbndbzsbzifile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.discharge_points`
     - Discharge source terms.
     - ``srcfile``, ``disfile``, ``netsrcdisfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.precipitation`
     - Spatially or temporally variable rainfall.
     - ``precipfile``, ``netamprfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.wind`
     - Wind forcing data.
     - ``wndfile``, ``netamuvfile``
   * - :py:attr:`~hydromt_sfincs.SfincsModel.pressure`
     - Atmospheric pressure fields.
     - ``netamrfile``


Please be aware that the indexfile is not included in the grid dataset.
Instead, it is generated during the writing process based on the mskfile,
and it is utilized for the purpose of reading grid variables.

.. currentmodule:: hydromt_sfincs.sfincs

.. _model_methods:

Model create methods
====================

An overview of the available SFINCS model create methods is provided in the table below.
When using HydroMT from the command line, only the create methods are exposed. Click on
a specific method see its documentation.

General create methods
---------------------

.. _general_setup_table:

.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.config.update`
     - Update SFINCS config (sfincs.inp) with a dictionary.
   * - :py:func:`~hydromt_sfincs.SfincsModel.grid.create`
     - This component generates a user-defined model grid.
   * - :py:func:`~hydromt_sfincs.SfincsModel.grid.create_from_region`
     - This component automatically generates a model grid covering the region of interest with a given resolution.

Grid create methods
------------------

.. _grid_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.elevation.create`
     - This component interpolates topobathy (depfile) data to the model grid.
   * - :py:func:`~hydromt_sfincs.SfincsModel.mask.create_active`
     - This component generates a mask (mskfile) defining which part of the model grid is active based on elevation criteria and/or polygons.
   * - :py:func:`~hydromt_sfincs.SfincsModel.mask.create_boundary`
     - This component adds boundary cells in the model mask (mskfile) based on elevation criteria and/or polygons.
   * - :py:func:`~hydromt_sfincs.SfincsModel.rivers.create_inflow`
     - This component adds boundary cells in the model mask (mskfile) where a river flows out of the model domain.
   * - :py:func:`~hydromt_sfincs.SfincsModel.roughness.create`
     - This component adds a Manning roughness map (manningfile) to the model grid based on gridded Manning roughness data or a
       combinataion of gridded land-use/land-cover map and a Manning roughness mapping table.
   * - :py:func:`~hydromt_sfincs.SfincsModel.infiltration.create_constant`
     - This component adds a spatially varying constant infiltration rate map (qinffile) to the model grid.
   * - :py:func:`~hydromt_sfincs.SfincsModel.infiltration.create_cn`
     - This component adds a potential maximum soil moisture retention map (scsfile) to the model grid based on a gridded curve number map.
   * - :py:func:`~hydromt_sfincs.SfincsModel.infiltration.create_cn_with_recovery`
     - This component adds a three layers related to the curve number (maximum and effective infiltration capacity; seff and smax) and
       saturated hydraulic conductivity (ks, to account for recovery) to the model
       grid based on landcover, Hydrological Similarity Group and saturated hydraulic conductivity (Ksat).
   * - :py:func:`~hydromt_sfincs.SfincsModel.storage_volume.create`
     - This component adds a storage volume map (volfile) to the model grid to account for green-infrastructure.
   * - :py:func:`~hydromt_sfincs.SfincsModel.subgrid.create`
     - This component generates subgrid tables (sbgfile) for the model grid based on a list of elevation and Manning roughness datasets

Geoms setup methods
-------------------

.. _geoms_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.observation_points.create`
     - This component adds observation points to the model (obsfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.cross_sections.create`
     - This component adds cross-sections to the model (crsfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.thin_dams.create`
     - This component adds line element structures to the model (thdfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.weirs.create`
     - This component adds line element structures to the model (weirfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.drainage_structures.create`
     - This component adds drainage structures (pump, culvert, one-way-valve) to the model (drnfile).

Forcing setup methods
---------------------

.. _forcing_setup_table:
.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_waterlevel_bnd_from_mask`
     - This component adds waterlevel boundary points (bndfile) along model waterlevel boundary (msk=2).
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_waterlevel_forcing`
     - This component adds waterlevel forcing (bndfile, bzsfile) from a `geodataset` (geospatial point timeseries) or a tabular `timeseries` dataframe.
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

Other setup methods
-------------------

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
