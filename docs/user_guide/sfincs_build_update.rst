.. _sfincs_build_update:

Building or updating a model
====================================

This plugin allows users to build or update a SFINCS model from available data. 
For beginning users, we recommend to use the :ref:`command line interface <sfincs_cli>` to build or update a model. 
In case you want to (locally) modify the model input data before generating a model, we recommend to use the :ref:`Python scripting <sfincs_python>`. 

In the following sections, examples are provided in iPython notebooks how to build your SFINCS model with HydroMT using either the CLI or python scripting.

.. _sfincs_cli:

Command Line Interface (CLI) - Basic
-------------------------------------

This plugin allows users to **build** a complete SFINCS model from available data for your area of interest. The model region_ is typically defined by a bounding box, 
see example below, or a geometry file. Once the configuration and data libraries are set, you can build a model by using: 

.. code-block:: console

    hydromt build sfincs path/to/built_model -r "{'bbox': [xmin, ymin, xmax, ymax]}" -i sfincs_build.yml -d data_sources.yml -vv

Or to **update**  an existing SFINCS model:

.. code-block:: console

    hydromt update sfincs ./sfincs_compound -o ./sfincs_compound_precip -i sfincs_update_precip.yml -vv

**Configuration file:**

Settings to build or update a SFINCS model are managed in a configuration file. In this file,
every option from each :ref:`model component <model_methods>` can be changed by the user
in its corresponding section. See the HydroMT core documentation for more info about the `model configuration .yml-file <config>`_

Note that the order in which the components are listed in the yml-file is important (methods are executed from top to bottom): 

- :py:func:`~hydromt_sfincs.SfincsModel.setup_grid` or :py:func:`~hydromt_sfincs.SfincsModel.setup_grid_from_region` should always be run first to define the model grid
- a lot of methods need elevation data to work properly, so :py:func:`~hydromt_sfincs.SfincsModel.setup_dep` should be run before most other methods 
- if discharge locations are inferred from hydrography, :py:func:`~hydromt_sfincs.SfincsModel.setup_river_inflow` should be run before :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing` or :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing_from_grid`.

**Data libraries:**

Data sources in HydroMT are provided in one of several yaml libraries. These libraries contain required
information on the different data sources so that HydroMT can process them for the different models. There
are three ways for the user to select which data libraries to use:

- If no yaml file is selected, HydroMT will use the data stored in the
  `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_
  which contains an extract of global data for a small region around the Piave river in Northern Italy.
- Another options for Deltares users is to select the deltares-data library (requires access to the Deltares
  P-drive). In the command line interface, this is done by adding either **-dd** or **--deltares-data**
  to the build / update command line.
- Finally, the user can prepare its own yaml libary (or libraries) (see
  `HydroMT documentation <https://deltares.github.io/hydromt/latest/index>`_ to check the guidelines).
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **yml file**
  with the **data_libs** option in the  `global` section.

See :ref:`Build simple SFINCS model from CLI <sfincs_compound>` for suggested components
and options to use for compound flooding applications.

.. _sfincs_python:

Python scripting - Advanced
------------------------------

Next to the command line interface, HydroMT-SFINCS also allows to setup (or interact with) a SFINCS model from Python scripting. 
The main advantage of this approach is that you can work with in-memory datasets, e.g. datasets that you have modified, next to datasets that are defined in the data catalog.

Typical applications where this approach can be useful are:
- when you want to modify gridded data (e.g. elevation or manning) before creating a model
- when you want to modify the forcing conditions (e.g. discharge or precipitation) while creating multiple scenarios 
- when you want to remove one of the forcing locations (e.g. a river inflow point) from the model

.. code-block:: python

    from hydromt_sfincs import SfincsModel
  
    sf = SfincsModel(data_libs=["artifact_data"], root="sfincs_compound")

    sf.setup_grid(x0=318650, y0=5040000, dx=50.0, dy=50.0, nmax=107, mmax=250, rotation=27, epsg=32633)

    # retrieve GEBCO elevation data from data catalog
    da = sf.data_catalog.get_rasterdataset("gebco", geom=sf.region, buffer=5)

    # modify elevation data by adding 1 m
    da = da + 1

    # use modifed (in-memory) elevation data to create model
    sf.setup_dep(datasets_dep=[{"da":da}])

    sf.plot_basemap()

    sf.write() # write all


.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
.. _config: https://deltares.github.io/hydromt/latest/user_guide/model_config.html

.. toctree::
    :hidden:
    
    Example: Build from CLI <../_examples/build_from_cli.ipynb>
    Example: Build from Python <../_examples/build_from_script.ipynb>
