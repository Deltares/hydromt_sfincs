.. _sfincs_build:

Building or update a model
==========================

This plugin allows users to **build** a complete SFINCS model from available data: 

.. code-block:: console

    hydromt build sfincs path/to/built_model "{'bbox': [xmin, ymin, xmax, ymax]}" -i sfincs_build.ini -d data_sources.yml -vv

Or to **update**  an existing SFINCS model:

.. code-block:: console

    hydromt update sfincs ./sfincs_coastal -o ./sfincs_coastal_precip -i sfincs_update_precip.ini -vv

.. _sfincs_config:

Configuration file
------------------
Settings to build or update a Wflow model are managed in a configuration file. In this file,
every option from each :ref:`model component <model_methods>` can be changed by the user
in its corresponding section. See the HydroMT core documentation for more info about the `model configuration .ini file <config>`_

Note that the order in which the components are listed in the ini file is important: 

- ``setup_region`` and then ``setup_topobathy`` should always be run first to define the model grid
- if discharge location are inferred from hydrography, ``setup_river_inflow`` should be run before ``setup_q_forcing`` or ``setup_q_forcing_from_grid``.

See :ref:`Coastal SFINCS model schematization <sfincs_coastal>` and 
:ref:`Riverine SFINCS model schematization <sfincs_riverine>` for suggested components
and options to use for coastal or riverine applications.

Selecting data
--------------
Data sources in HydroMT are provided in one of more yaml data catalog files. 
Checkout the HydroMT core documentation for more info on `working with data in HydroMT <data>`_.

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_overview.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
.. _config: https://deltares.github.io/hydromt/latest/user_guide/model_config.html


.. toctree::
    :hidden:
    
    sfincs_coastal.rst
    sfincs_riverine.rst
    Example: Build a coastal SFINCS model <../_examples/build_coastal_model.ipynb>
    Example: Build a riverine SFINCS model <../_examples/build_riverine_model.ipynb>
    Example: Build a SFINCS model from Python <../_examples/build_from_py.ipynb>
    Example: Update SFINCS model components <../_examples/update_model.ipynb>