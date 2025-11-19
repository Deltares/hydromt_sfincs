.. _model_update:

=======================
Updating a SFINCS model
=======================

Command Line Interface (CLI) - Basic
=====================================

This plugin allows users to **update** an existing SFINCS model from available data.
Once the configuration and data libraries are set, you can update a model by using:

.. code-block:: console

    hydromt update sfincs ./sfincs_compound -o ./sfincs_compound_precip -i sfincs_update_precip.yml -vv

Configuration file
-------------------

For updating a model, you can use a configuration file similar to the one below.

.. code-block:: yaml

  global:
    data_libs:                          # add optional paths to data_catalog yml files
    - artifact_data

  steps:
    - config.update:
        tref: "20100201 000000"
        tstart: "20100201 000000"
        tstop: "20100202 000000"
    - precipitation.create:
        precip: "era5_hourly"

Python scripting - Advanced
===========================

HydroMT-SFINCS also allows to setup (or interact with) a SFINCS model from Python scripting.
The main advantage of this approach is that you can work with in-memory datasets, e.g. datasets that you have modified,
next to datasets that are defined in the data catalog.

Important when updating a SfincsModel from python scripting is that you initialze an existing model in
the right mode, for updating that would be "r+" (append). If you don't want to overwrite an existing model,
you could open the model in read-only mode ("r") and write the updated model to a new location by changing the mode,
see example below.

.. code-block:: python

    from hydromt_sfincs import SfincsModel
    from datetime import datetime

    # open existing model in read-only mode
    sf = SfincsModel(data_libs=["artifact_data"], root="sfincs_compound", mode="r")
    # read the model
    sf.read()

    # change the root and mode to write the updated model to a new location
    sf.root.set("sfincs_compound_precip", mode="w+")

    # update the configuration
    sf.config.update(
        {"tref": datetime(2010, 2, 1, 0, 0, 0),
        "tstart": datetime(2010, 2, 1, 0, 0, 0),
        "tstop": datetime(2010, 2, 2, 0, 0, 0)}
    )

    # update the precipitation component
    sf.precipitation.create(precip="era5_hourly")

    # write the model
    sf.write()

Examples
--------

For examples of how to update a SFINCS model using HydroMT-SFINCS, see:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    Example: Update Forcing <../../_examples/2_update_forcing.ipynb>
    Example: Update Geometries <../../_examples/3_update_geometries.ipynb>
