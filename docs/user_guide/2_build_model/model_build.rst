.. _model_build:

=======================
Building a SFINCS model
=======================

This plugin allows users to build a SFINCS model from available data.
There are 2 main ways to use HydroMT to build your SFINCS model:

**1. Command Line Interface (basic user)**:
      Provide some information about the model configuration in a .yml-file and quickly build the model using the Command Line Interface (CLI).
      The .yml-file provides a way to create a reproducible model setup recipe, which can be easily be shared with others.
      Additionally, no Python knowledge is required to use the CLI.

**2. Python scripting (advanced user)**:
      Dive into the underlying Python functions and use those to build your model from scratch in a Python script.
      This option is recommended for the expert user who wants to (locally) adjust the model input data as part of the model building process,
      e.g. in the case of in-memory modifications of the bed levels or variations on the boundary conditions.
      The Python interface provides a lot of flexibility and access to the full HydroMT-SFINCS API, but requires some knowledge of Python.

In the following sections, examples are provided how to build your SFINCS model with HydroMT using either the CLI or Python scripting.

.. _sfincs_cli:

Command Line Interface (CLI) - Basic
====================================

This plugin allows users to **build** a complete SFINCS model from available data for your area of interest.
Once the configuration and data libraries are set, you can build a model by using:

.. code-block:: console

    hydromt build sfincs path/to/built_model -i sfincs_build.yml -d data_sources.yml -vv

.. note::
    From HydroMT version 1.0 onwards, the region argument has been moved to
    grid create methods and is no longer available via cli.

Configuration file
-------------------

Settings to build a SFINCS model are managed in a configuration file. In this file,
every option from each :ref:`model method <model_methods>` can be changed by the user
in its corresponding section. See the HydroMT core documentation for more info about the
`model configuration .yml-file <config>`_ and check-out the example below.

.. code-block:: yaml

  global:
    data_libs:                          # add optional paths to data_catalog yml files
    - artifact_data

  steps:
    - config.update:
        tref: "20100201 000000"
        tstart: "20100201 000000"
        tstop: "20100202 000000"
    - grid.create_from_region:
        region:                         # define model region
          geom: region                  # Note that this is defined in the data catalog
        res: 50                         # model resolution
        crs: utm                        # model CRS (must be UTM zone)
        rotated: True                   # allow a rotated grid
    - elevation.create:
        elevation_list:
        - elevation: merit_hydro        # 1st elevation dataset
          zmin: 0.001                   # only use where values > 0.001
        - elevation: gebco              # 2nd eleveation dataset (to be merged with the first)
    - mask.create_active:
        mask: data//region.geojson      # Note that this is local data and only valid for this example
        zmin: -5                        # set cells with an elevation <-5 to inactive
    - mask.create_boundary:
        btype: waterlevel               # Set waterlevel boundaries
        zmax: -1                        # only cells with an elevation <-1 can be waterlevel boundaries

.. note::
    The order in which the components are listed in the yml-file is important (methods are executed from top to bottom):

- :py:func:`~hydromt_sfincs.SfincsModel.grid.create` or :py:func:`~hydromt_sfincs.SfincsModel.grid.create_from_region` should always be run first to define the model grid.
- Many methods (e.g., :py:func:`~hydromt_sfincs.SfincsModel.mask.create_active`) need elevation data to work properly, hence :py:func:`~hydromt_sfincs.SfincsModel.elevation.create` should be run before most other methods.
- If discharge locations are inferred from hydrography, :py:func:`~hydromt_sfincs.SfincsModel.rivers.create_inflow` should be run before :py:func:`~hydromt_sfincs.SfincsModel.discharge_points.create`.
- If water level bounary points are inferred from the water level mask cells, :py:func:`~hydromt_sfincs.SfincsModel.water_level.create_boundary_points_from_mask` should be run before :py:func:`~hydromt_sfincs.SfincsModel.water_level.create`.

Example
--------

See `Example: Build from CLI <../../_examples/build_from_cli.ipynb>`_ for suggested components
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

    sf.grid.create(x0=318650, y0=5040000, dx=50.0, dy=50.0, nmax=107, mmax=250, rotation=27, epsg=32633)

    # retrieve GEBCO elevation data from data catalog
    da = sf.data_catalog.get_rasterdataset("gebco", geom=sf.region, buffer=5)

    # modify elevation data by adding 1 m
    da = da + 1

    # use modifed (in-memory) elevation data to create model
    sf.elevation.create(elevation_list=[{"da":da}])

    sf.plot_basemap()

    sf.write() # write all

Example
--------

See examples below for more detailed examples:

.. toctree::
   :maxdepth: 2
   :titlesonly:

    Example: Build from CLI <../../_examples/0_build_from_cli.ipynb>
    Example: Build from script <../../_examples/1_build_from_script.ipynb>

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
.. _config: https://deltares.github.io/hydromt/latest/user_guide/model_config.html
