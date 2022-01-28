.. _sfincs_riverine:

===========================
Setting up a riverine model
===========================

To build a **riverine flood model** with q (discharge) and p (precipitation), but no h 
(waterlevel) boundary conditions, the model region_ can either be defined by a 
bounding or the interbasin within a bounding box. 

In the second example below the region is based on the interbasin is based on all cells 
within the bounding box draining into a stream with a minimal upstream area of 20 km2. 
The delineation is based on gridded flow direction data set in the setup_region method. 

At the upstream end of each river a discharge source point can be defined with the 
`setup_river_inflow` method and at the downstream end an  outflow boundary with the 
`setup_river_outflow` method.

As DEMs typically to not have an accurate representation of the river underwater 
bathymetry, this can be 'burned' into the DEM using the `setup_river_bathymetry` method.
Note that this method requires `setup_river_hydrography` to be executed first.

A typical workflow to setup a riverine model schematization is provided in the
:download:`sfincs_riverine.ini <../_examples/sfincs_riverine.ini>` and shown below. 
Each section corresponds to one of the model :ref:`components` and the `[global]` section can be used to pass
additional arguments to the :py:class:`~hydromt.models.sfincs.SfincsModel`. initialization. 


.. code-block:: console

    hydromt build sfincs /path/to/model_root "{'bbox': [xmin, ymin, xmax, ymax]}" -i sfincs_riverine.ini -vv

.. code-block:: console

    hydromt build sfincs /path/to/model_root "{'interbasin': [xmin, ymin, xmax, ymax], 'uparea': 20}" -i sfincs_riverine.ini -vv


.. literalinclude:: ../_examples/sfincs_riverine.ini
   :language: Ini


.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options