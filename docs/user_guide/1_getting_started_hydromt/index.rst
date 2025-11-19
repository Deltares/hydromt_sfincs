Getting started with HydroMT
================================

The core functionalities of HydroMT, such as the command-line interface (CLI), data catalog management, and region definitions, are documented in the main HydroMT documentation.
These topics are shared across all HydroMT model plugins (including HydroMT-SFINCS) and are maintained in one central location to ensure consistency.
Refer to the sections below for the official and up-to-date information.

For the users of the HydroMT-SFINCS plugin, the data catalog and model region concepts are particularly important, as they form the basis for building and managing SFINCS models using HydroMT.

.. grid:: 2
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: data_catalog
        :link-type: ref

        :octicon:`database;5em;sd-text-icon blue-icon`
        +++
        Data Catalog
        Understand how HydroMT manages data sources and integrates external datasets.

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt/stable/guides/user_guide/model_region.html
        :link-type: url

        :octicon:`globe;5em;sd-text-icon blue-icon`
        +++
        Model Region
        Learn how to define, configure, and manage model regions within HydroMT.

.. toctree::
   :hidden:
   :caption: HydroMT Core
   :maxdepth: 2

   Data Catalog <data_catalog>
   Model Region <https://deltares.github.io/hydromt/stable/guides/user_guide/model_region.html>
