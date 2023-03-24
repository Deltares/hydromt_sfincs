.. currentmodule:: hydromt_sfincs.sfincs

============================
Model components
============================

When making a SFINCS model, you need to create multiple input files.
With the HydroMT SFINCS plugin, you can easily make these SFINCS model schematizations. 
This plugin helps you preparing or updating several model components of a SFINCS model 
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and 
discharge forcing.

.. _model_components:

The following table provides an overview of which :py:class:`~hydromt_sfincs.SfincsModel` 
model data component (attribute) contains which SFINCS in- and output files. 

For more information about what each individual file is, see the `SFINCS documentation <https://sfincs.readthedocs.io/en/latest/>`_.

.. list-table:: SfincsModel data component
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_sfincs.SfincsModel` attribute
     - SFINCS files
   * - :py:attr:`~hydromt_sfincs.SfincsModel.config`
     - sfincs.inp
   * - :py:attr:`~hydromt_sfincs.SfincsModel.staticmaps`
     - depfile, mskfile, indexfile, sbgfile, manningfile, qinffile, scsfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.staticgeoms`
     - obsfile, thdfile, weirfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.forcing`
     - bndfile, bzsfile, srcfile, disfile, precipfile, netamprfile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.states`
     - inifile
   * - :py:attr:`~hydromt_sfincs.SfincsModel.results`
     - sfincs_his.nc, sfincs_map.nc

Note that the indfile is not part of the staticmaps dataset but created based on 
the mskfile upon writing and used for reading staticmaps.

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
