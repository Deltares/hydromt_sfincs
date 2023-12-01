.. currentmodule:: hydromt_sfincs.sfincs

============================
Model components and methods
============================

When making a SFINCS model, you need to create multiple input files.
With the HydroMT SFINCS plugin, you can easily make these SFINCS model schematizations. 
This plugin helps you preparing or updating several model components of a SFINCS model 
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and 
discharge forcing.

.. _model_components:

Model components
================

The following table provides an overview of which :py:class:`~hydromt_sfincs.SfincsModel` 
model data component (attribute) contains which SFINCS in- and output files. The files are read and written with the associated
read- and write- methods, i.e. :py:func:`~hydromt_sfincs.SfincsModel.read_config`
and :py:func:`~hydromt_sfincs.SfincsModel.write_config` for the
:py:attr:`~hydromt_sfincs.SfincsModel.config` component.

For more information about each file, see the `SFINCS documentation <https://sfincs.readthedocs.io/en/latest/>`_.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` attribute
     - SFINCS files
   * - :py:attr:`~hydromt_sfincs.SfincsModel.config`
     - sfincs.inp
   * - :py:attr:`~hydromt_sfincs.SfincsModel.grid`
     - depfile, mskfile, indexfile, manningfile, qinffile, scsfile, smaxfile, sefffile, krfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.subgrid`
     - sbgfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.geoms`
     - obsfile, thdfile, weirfile, drnfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.forcing`
     - bndfile, bzsfile, srcfile, disfile, precipfile, netbndbzsbzifile, netsrcdisfile, netamprfile, netampfile, netamuamvfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.states`
     - inifile, rstfile (not yet implemented)
   * - :py:attr:`~hydromt_sfincs.SfincsModel.results`
     - sfincs_his.nc, sfincs_map.nc

Please be aware that the indexfile is not included in the grid dataset. 
Instead, it is generated during the writing process based on the mskfile, 
and it is utilized for the purpose of reading grid variables.

.. currentmodule:: hydromt_sfincs.sfincs

.. _model_methods:

Model setup methods
====================

An overview of the available SFINCS model setup methods is provided in the table below. 
When using HydroMT from the command line, only the setup methods are exposed. Click on
a specific method see its documentation.

General setup methods
---------------------

.. _general_setup_table:

.. list-table::
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_config`     
     - Update SFINCS config (sfincs.inp) with a dictionary.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_grid`
     - This component generates a user-defined model grid.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_grid_from_region`
     - This component automatically generates a model grid covering the region of interest with a given resolution.

Grid setup methods
------------------

.. _grid_setup_table:
.. list-table:: 
   :widths: 20 55
   :header-rows: 1
   :stub-columns: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` Method
     - Explanation     
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_dep`
     - This component interpolates topobathy (depfile) data to the model grid.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_active`
     - This component generates a mask (mskfile) defining which part of the model grid is active based on elevation criteria and/or polygons.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_bounds`
     - This component adds boundary cells in the model mask (mskfile) based on elevation criteria and/or polygons.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_river_outflow`
     - This component adds boundary cells in the model mask (mskfile) where a river flows out of the model domain.   
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_manning_roughness`
     - This component adds a Manning roughness map (manningfile) to the model grid based on gridded Manning roughness data or a combinataion of gridded land-use/land-cover map and a Manning roughness mapping table.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_constant_infiltration`
     - This component adds a spatially varying constant infiltration rate map (qinffile) to the model grid.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_cn_infiltration`
     - This component adds a potential maximum soil moisture retention map (scsfile) to the model grid based on a gridded curve number map.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_cn_infiltration_with_ks`
     - This component adds a three layers related to the curve number (maximum and effective infiltration capacity; seff and smax) and saturated hydraulic conductivity (ks, to account for recovery) to the model 
     grid based on landcover, Hydrological Similarity Group and saturated hydraulic conductivity (Ksat).
   * - :py:funct:`~hydromt_sfincs.SfincsModel.setup_storage_volume~
     - This component adds a storage volume map (volfile) to the model grid to account for green-infrastructure.
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_subgrid`
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
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_observation_points`
     - This component adds observation points to the model (obsfile). 
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_structures`
     - This component adds line element structures to the model (thdfile, weirfile).
   * - :py:func:`~hydromt_sfincs.SfincsModel.setup_drainage_structures`
     - This component adds drainage structures (pump, culvert) to the model (drnfile).     

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
