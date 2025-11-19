.. _intro_user_guide:

User guide
==========

With the **HydroMT SFINCS plugin**, users can easily benefit from the rich set of tools of the
`HydroMT package <https://github.com/Deltares/hydromt>`_ to build and update
`SFINCS <https://sfincs.readthedocs.io/en/latest/>`_ models from available global and local data.
The user guide is structured in the following way:

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: 1_getting_started_hydromt/index
        :link-type: doc

        :octicon:`rocket;5em;sd-text-icon blue-icon`
        +++
        Getting started with HydroMT

    .. grid-item-card::
        :text-align: center
        :link: 2_build_model/index
        :link-type: doc

        :octicon:`container;5em;sd-text-icon blue-icon`
        +++
        Building a SFINCS model

    .. grid-item-card::
        :text-align: center
        :link: 3_update_model/index
        :link-type: doc

        :octicon:`pencil;5em;sd-text-icon blue-icon`
        +++
        Updating a SFINCS model

    .. grid-item-card::
        :text-align: center
        :link: 4_run_model/sfincs_run
        :link-type: doc

        :octicon:`gear;5em;sd-text-icon blue-icon`
        +++
        Running a SFINCS model

    .. grid-item-card::
        :text-align: center
        :link: 5_postprocess_model/sfincs_analyse
        :link-type: doc

        :octicon:`graph;5em;sd-text-icon blue-icon`
        +++
        Processing and Visualization

    .. grid-item-card::
        :text-align: center
        :link: 6_migration_guide/migration_hydromt
        :link-type: doc

        :octicon:`arrow-switch;5em;sd-text-icon blue-icon`
        +++
        Migration Guide

This plugin assists the SFINCS modeller in:

- Quickly setting up a base SFINCS model and default parameter values.
- Making maximum use of the best available global or local data.
- Adjusting and updating components of a SFINCS model and their associated parameters in a consistent way.
- Connecting SFINCS to other models (input from e.g. Wflow, output towards e.g. Delft-FIAT)
- Visualizing SFINCS models.
- Analysing SFINCS model outputs.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2
   :hidden:

   1_getting_started_hydromt/index.rst
   2_build_model/index.rst
   3_update_model/index.rst
   4_run_model/sfincs_run.rst
   5_postprocess_model/sfincs_analyse.rst
   6_migration_guide/migration_hydromt.rst
