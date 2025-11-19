.. _migration_hydromt_v1:

Migration Guide
===============

HydroMT v1 introduces a component-based architecture to replace the previous inheritance model.
Instead of all model functionality being defined in a single ``Model`` class, a model is now composed of modular ``ModelComponent``
classes such as ``GridComponent`` or ``ConfigComponent``.
This structure makes models more flexible, extensible, and easier to maintain.
For detailed guidance, refer to the official `HydroMT migration guide <https://deltares.github.io/hydromt/latest/user_guide/migration_guide/index.html>`_.

The component-based architecture of HydroMT v1 allows for better separation of functionality. Where previously all functionality was described as part of the
`SfincsModel` class (in sfincs.py), the new architecture allows to break this down into dedicated components that each handle a specific part of the model.
For example, the configuration is now handled by a `SfincsConfigComponent` described in components/config.py. This resulted in a mayor code restructure of HydroMT-SFINCS,
but the overall functionality and API remains very similar earlier versions. In these migration guidelines,
we highlight the main changes and provide guidance on how to upgrade existing HydroMT-SFINCS workflows to be compatible with HydroMT v1.


Data Catalog Format Changes
---------------------------

Information
^^^^^^^^^^^
The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (`DataSource`, `Driver`, `URIResolver`, and `DataAdapter`).

Key format changes:

- ``path`` renamed to ``uri``
- ``filesystem`` or ``driver_kwargs`` moved under ``driver``
- ``unit_add``, ``unit_mult``, ``rename``, etc. moved under ``data_adapter``
- ``crs`` and ``nodata`` moved under ``metadata`` (renamed from ``meta``)
- A single catalog entry can now reference multiple data variants or versions

For detailed information about the format changes, see this section in the HydroMT migration guide: `Changes to the data catalog yaml file format <https://deltares.github.io/hydromt/latest/user_guide/migration_guide/data_catalog.html>`_

How to upgrade
^^^^^^^^^^^^^^
All existing pre-defined catalogs have been updated to the new format. For your own catalogs, you can upgrade
easily with the HydroMT ``check`` command:

.. code-block:: bash

   hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v

.. note::

   When an exclamation mark (!) is put before a command line code, the command is to be executed in a Jupyter notebook cell.


Main Changes for Users
----------------------

In HydroMT-SFINCS v2, the internal data structure and API were redesigned to improve consistency and maintainability.
Most changes affect how model components (such as ``grid`` and ``forcing``) are accessed and how model data is read and written.

Component Changes
^^^^^^^^^^^^^^^^^

In earlier versions of HydroMT-SFINCS, all model functionality was encapsulated within the `SfincsModel` class. The data to describe the model
was stored in raw data structures such as `xarray.Dataset`, `dict`, or `geopandas.GeoDataFrame`, and could be accessed directly via attributes of the `SfincsModel` instance.
These attributes were grouped into `config`, `grid`, `geoms`, `forcing`, and `results`. The new component-based architecture allows to further separate these model parts
into dedicated classes that each handle a specific part of the model. The table below summarizes the mapping from old attributes to new component names and classes.

+---------------------+--------------------------+------------------------------------------+
| Old Attribute       | New Component Name       | New Class                                |
+=====================+==========================+==========================================+
| config              | config                   | SfincsConfig                             |
+---------------------+--------------------------+------------------------------------------+
| grid                | grid                     | SfincsGrid                               |
|                     | elevation                | SfincsElevation                          |
|                     | mask                     | SfincsMask                               |
|                     | infiltration             | SfincsInfiltration                       |
|                     | roughness                | SfincsRoughness                          |
|                     | storage_volume           | SfincsStorageVolume                      |
|                     | initial_conditions       | SfincsInitialConditions                  |
|                     | subgrid                  | SfincsSubgridTable                       |
+---------------------+--------------------------+------------------------------------------+
| geoms               | observation_points       | SfincsObservationPoints                  |
|                     | cross_sections           | SfincsCrossSections                      |
|                     | thin_dams                | SfincsThinDams                           |
|                     | weirs                    | SfincsWeirs                              |
|                     | wave_makers              | SfincsWaveMakers                         |
|                     | drainage_structures      | SfincsDrainageStructures                 |
+---------------------+--------------------------+------------------------------------------+
| forcing             | water_level              | SfincsWaterLevel                         |
|                     | discharge_points         | SfincsDischargePoints                    |
|                     | precipitation            | SfincsPrecipitation                      |
|                     | pressure                 | SfincsPressure                           |
|                     | wind                     | SfincsWind                               |
|                     | rivers                   | SfincsRivers                             |
+---------------------+--------------------------+------------------------------------------+
| results             | output                   | SfincsOutput                             |
+---------------------+--------------------------+------------------------------------------+

Method Changes
^^^^^^^^^^^^^^

The model components are now **dedicated classes** rather than raw data objects (e.g., ``xarray``, ``dict``, or ``geopandas``).
This means that the way to access and manipulate model data has changed. Each component can be accessed via the ``model`` instance
and exposes its underlying methods (e.g., ``read()``, ``write()``, ``create()`` and ``set()``).
The actual data is now accessed via the ``.data`` property of each component.
The table below summarizes the main method changes from HydroMT-SFINCS v1.x to v2.

+--------------------------------+--------------------------------+
| v1.x                           | v2                             |
+================================+================================+
| ``model.<component>``          | ``model.<component>.data``     |
+--------------------------------+--------------------------------+
| ``model.write_<component>()``  | ``model.<component>.write()``  |
+--------------------------------+--------------------------------+
| ``model.read_<component>()``   | ``model.<component>.read()``   |
+--------------------------------+--------------------------------+
| ``model.set_<component>()``    | ``model.<component>.set()``    |
+--------------------------------+--------------------------------+


Probably the most significant change is the way we "create" or "update" components. In HydroMT-SFINCS v1.x, this was done via "setuo" methods on the main model class,
such as ``model.setup_grid()`` or ``model.setup_<component>()``. In HydroMT-SFINCS v2, this is now done via the ``create()`` method on each component.
For a complete conversion table of method changes, see below:

.. csv-table:: Conversion Table
   :header: "v1.x","v2.x","Argument Mapping"
   :widths: 20,20,40
   :file: conversion_table.csv

Example: Accessing Component Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each component provides structured access to its data via the ``.data`` property.

.. code-block:: python

    from hydromt_sfincs import SfincsModel

    model = SfincsModel(root="path/to/model", mode="r")

    # Access xarray.Dataset of the grid component
    grid = model.grid.data

    # Access geometries (GeoDataFrames), such as the observation points
    observation_points = model.observation_points.data

    # Access forcing data (xarray.Dataset), such as water level time series
    water_level = model.water_level.data

Example: Writing Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read and write operations are now handled at the **component level**.

.. code-block:: python

    # Write configuration file
    model.config.write()

    # Write updated grid to disk
    model.grid.write()

These changes provide a clearer and more modular interface, making it easier to manipulate model components independently.


YAML Configuration Changes
--------------------------

The HydroMT model configuration format has been overhauled and the ini format is not supported anymore.
The root YAML file now includes three main keys: ``modeltype``, ``global``, and ``steps``.

- ``modeltype`` (optional): Defines which model plugin is being used (e.g. ``sfincs``).
- ``global``: Defines model-wide configuration, including data catalog(s), name of the model configuration toml file etc.
- ``steps``: Replaces the old numbered dictionary format with a sequential list of function calls.

Some of the functions (component specific read and write) are now explicitly mapped to model or component methods using the `<component>.<method>` syntax.

For a complete example of the new configuration format, see the SFINCS v1 YAML template: :download:`sfincs_build.yml </_examples/sfincs_base_build.yml>`.

For more information on the format changes, see this section in the HydroMT migration guide: `Changes to the yaml HydroMT configuration file format <https://deltares.github.io/hydromt/latest/user_guide/migration_guide/model_workflow.html>`_.
