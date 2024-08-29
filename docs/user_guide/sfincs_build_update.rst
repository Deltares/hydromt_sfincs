.. _sfincs_build_update:

=============================
Building or updating a model
=============================

This plugin allows users to build or update a SFINCS model from available data using
the :ref:`command line interface <sfincs_cli>` or :ref:`Python scripting <sfincs_python>`.
For a brief overview of the differences, see :ref:`Working with the SFINCS model <working_with_sfincs>`.

In the following sections, examples are provided how to build your SFINCS model with HydroMT using either the CLI or Python scripting.

.. _sfincs_cli:

Command Line Interface (CLI) - Basic
=====================================

This plugin allows users to **build** a complete SFINCS model from available data for your area of interest. The model region_ is typically defined by a bounding box,
see example below, or a geometry file. Once the configuration and data libraries are set, you can build a model by using:

.. code-block:: console

    hydromt build sfincs path/to/built_model -r "{'bbox': [xmin, ymin, xmax, ymax]}" -i sfincs_build.yml -d data_sources.yml -vv

Or to **update**  an existing SFINCS model:

.. code-block:: console

    hydromt update sfincs ./sfincs_compound -o ./sfincs_compound_precip -i sfincs_update_precip.yml -vv

Configuration file
-------------------

Settings to build or update a SFINCS model are managed in a configuration file. In this file,
every option from each :ref:`model method <model_methods>` can be changed by the user
in its corresponding section. See the HydroMT core documentation for more info about the `model configuration .yml-file <config>`_ and check-out the example below.

.. code-block:: yaml

  global:
    data_libs: []               # add optional paths to data_catalog yml files

  setup_config:
    tref: "20100201 000000"
    tstart: "20100201 000000"
    tstop: "20100202 000000"

  setup_grid_from_region:
    res: 50                     # model resolution
    crs: utm                    # model CRS (must be UTM zone)
    rotated: True               # allow a rotated grid

  setup_dep:
    datasets_dep:
    - elevtn: merit_hydro       # 1st elevation dataset
      zmin: 0.001               # only use where values > 0.001
    - elevtn: gebco             # 2nd eleveation dataset (to be merged with the first)

  setup_mask_active:
    mask: data//region.geojson  # Note that this is local data and only valid for this example
    zmin: -5                    # set cells with an elevation <-5 to inactive

  setup_mask_bounds:
    btype: waterlevel           # Set waterlevel boundaries
    zmax: -1                    # only cells with an elevation <-1 can be waterlevel boundaries

Note that the order in which the components are listed in the yml-file is important (methods are executed from top to bottom):

- :py:func:`~hydromt_sfincs.SfincsModel.setup_grid` or :py:func:`~hydromt_sfincs.SfincsModel.setup_grid_from_region` should always be run first to define the model grid.
- Many methods (e.g., :py:func:`~hydromt_sfincs.SfincsModel.setup_mask_active`) need elevation data to work properly, hence :py:func:`~hydromt_sfincs.SfincsModel.setup_dep` should be run before most other methods.
- If discharge locations are inferred from hydrography, :py:func:`~hydromt_sfincs.SfincsModel.setup_river_inflow` should be run before :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing` or :py:func:`~hydromt_sfincs.SfincsModel.setup_discharge_forcing_from_grid`.
- If water level bounary points are inferred from the water level mask cells,  :py:func:`~hydromt_sfincs.SfincsModel.setup_waterlevel_bnd_from_mask` should be run before :py:func:`~hydromt_sfincs.SfincsModel.setup_waterlevel_forcing`.

Data Catalogs
-------------

Data sources are provided to HydroMT in one or more user-definfed data catalog (yaml) files
or from pre-defined data catalogs. These data catalogs contain required information on the
different data sources so that HydroMT can process them for the different models.
There are three ways for the user to select which data catalog to use:

- There are several `pre-defined data catalog <https://deltares.github.io/hydromt/latest/user_guide/data_existing_cat.html>`_
  Amongst other, these include the `deltares_data` data catalog for Deltares users which requires access to the Deltares P-drive.
  More pre-defined data catalogs will be added in the future.
- Furthermore, the user can prepare its own yaml libary (or libraries) (see
  `HydroMT documentation <https://deltares.github.io/hydromt/latest/index>`_ to check the guidelines).
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **yml file**
  with the **data_libs** option in the  `global` section (see example above).
- Finally, if no catalog is provided, HydroMT will use the data stored in the
  `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_
  which contains an extract of global data for a small region around the Piave river in Northern Italy.

Example
--------

See `Example: Build from CLI <../_examples/build_from_cli.ipynb>`_ for suggested components
and options to use for compound flooding applications.

.. _sfincs_python:

Python scripting - Advanced
===========================

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

Example
--------

See `Example: Build from Script <../_examples/build_from_script.ipynb>`_ for a more detailed example.


.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
.. _config: https://deltares.github.io/hydromt/latest/user_guide/model_config.html

.. toctree::
    :hidden:

    Example: Build from CLI <../_examples/build_from_cli.ipynb>
    Example: Build from script <../_examples/build_from_script.ipynb>
    Example: Setup model forcing <../_examples/example_forcing.ipynb>
    .. Example: Working with data <../_examples/example_datasources.ipynb>
