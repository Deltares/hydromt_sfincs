.. _data_catalog:

Data Catalog
============

Data sources are provided to HydroMT in one or more user-definfed data catalog (yaml) files
or from pre-defined data catalogs. These data catalogs contain required information on the
different data sources so that HydroMT can process them for the different models.
There are three ways for the user to select which data catalog to use:

- There are several `pre-defined data catalog <https://deltares.github.io/hydromt/stable/guides/user_guide/data_existing_cat.html>`_
  Amongst other, these include the `deltares_data` data catalog for Deltares users which requires access to the Deltares P-drive.
  More pre-defined data catalogs will be added in the future.
- Furthermore, the user can prepare its own yaml libary (or libraries) (see
  `HydroMT documentation <https://deltares.github.io/hydromt/stable/index>`_ to check the guidelines).
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **yml file**
  with the **data_libs** option in the  `global` section (see example above).
- Finally, if no catalog is provided, HydroMT will use the data stored in the
  `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_
  which contains an extract of global data for a small region around the Piave river in Northern Italy.


The HydroMT core documentation provides comprehensive information on how to manage and extend data catalogs.
You can explore the guides below for detailed instructions and examples.

.. toctree::
   :hidden:
   :maxdepth: 1

   Prepare your data catalog <https://deltares.github.io/hydromt/stable/guides/advanced_user/data_prepare_cat.html>
   Supported data types <https://deltares.github.io/hydromt/stable/guides/advanced_user/data_types.html>
   Predefined catalogs <https://deltares.github.io/hydromt/stable/guides/user_guide/data_existing_cat.html>
