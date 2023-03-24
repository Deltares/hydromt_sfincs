.. currentmodule:: hydromt_sfincs.sfincs

=============
Model methods
=============
.. _model_methods:

With the HydroMT SFINCS plugin, you can easily work with SFINCS model schematizations. 
This plugin helps you preparing or updating several model components of a SFINCS model 
such as topography/bathymetry, roughness, infiltration maps and dynamic waterlevel and 
discharge forcing.

Available functions
===================

To make your model, different levels of functions are available to do this.
These are ordered in the code from high-level function to low-level:

- **1. Model setup_ method**: 
      General methods that are specified in the configuration file (e.g. [setup_grid_from_region]) - basic user

- **2. Model create_ function**: 
      More detailed Python methods to call when making your own Python script to build a model (e.g. sf.create_grid) - advanced user

- **3. Workflow methods**: 
      Lower level workflow methods that are called by the create functions - backend

- **4. SFINCS low-level methods**: 
      Lowest level functionality methods for e.g. only reading or writing files - backend

What type of function is important for you as user, depends whether you want to use HydroMT-SFINCS as a **basic user** from configuration file or **advanced user** from Python scripts.
Choose the subtab for what it appropriate for you, to find out more information about available methods and functions.

.. toctree::
    :hidden:
    
    sfincs_building_in_configuration.rst
    sfincs_building_in_python.rst