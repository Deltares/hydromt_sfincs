.. _sfincs_build:

Examples to build or update a model
==========================

In the additional tabs under this section, multiple examples are given in iPython notebooks how to build your SFINCS model in HydroMT using either a configuration file, or the underlying Python scripts.
For a brief overview, see the 2 options explained below.

From configuration file - Basic
-------------------------------

This plugin allows users to **build** a complete SFINCS model from available data: 

.. code-block:: console

    hydromt build sfincs path/to/built_model "{'bbox': [xmin, ymin, xmax, ymax]}" -i sfincs_build.ini -d data_sources.yml -vv

Or to **update**  an existing SFINCS model:

.. code-block:: console

    hydromt update sfincs ./sfincs_coastal -o ./sfincs_coastal_precip -i sfincs_update_precip.ini -vv

.. _sfincs_config:


Settings to build or update a SFINCS model are managed in a configuration file. In this file,
every option from each :ref:`model component <model_methods>` can be changed by the user
in its corresponding section. See the HydroMT core documentation for more info about the `model configuration .ini file <config>`_

Note that the order in which the components are listed in the ini file is important: 

- ``setup_region`` and then ``setup_topobathy`` should always be run first to define the model grid
- if discharge location are inferred from hydrography, ``setup_river_inflow`` should be run before ``setup_q_forcing`` or ``setup_q_forcing_from_grid``.

See :ref:`Coastal SFINCS model schematization <sfincs_coastal>` and 
:ref:`Riverine SFINCS model schematization <sfincs_riverine>` for suggested components
and options to use for coastal or riverine applications.

From Python scripts - Advanced
------------------------------

A short example of how these methods can be called in a separate Python script that you can build yourself is showed below:

.. code-block:: console

    from hydromt_sfincs import SfincsModel
  
    sf = SfincsModel(data_libs=["artifact_data"], root="sfincs_compound")

    sf.create_grid(grid_type="regular", **inp_dict)

    sf.plot_basemap()

    sf.write() # write all

Selecting data
--------------
Data sources in HydroMT are provided in one of more yaml data catalog files. 
Checkout the HydroMT core documentation for more info on `working with data in HydroMT <data>`_.

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
.. _config: https://deltares.github.io/hydromt/latest/user_guide/model_config.html

.. toctree::
    :hidden:
    
    sfincs_compound.rst
    Example: Build a simple compound SFINCS model from configuration file <../_examples/build_simple_compound_model.ipynb>

    Example: Build a simple compound SFINCS model from Python scripts <../_examples/build_simple_compound_model_from_script.ipynb>
    Example: Update a simple compound SFINCS model to subgrid from Python scripts <../_examples/upgrade_simple_compound_model_to_subgrid_from_script.ipynb>
    Example: Build an advanced compound SFINCS model from configuration file <../_examples/build_advanced_subgrid_compound_model_from_script.ipynb>