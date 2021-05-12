.. _sfincs_riverine:

===========================
Setting up a riverine model
===========================

To build a **riverine flood model** with q (discharge) and p (precipitation), but no h 
(waterlevel) boundary conditions, the model region_ is typically defined by the interbasin
within a bounding box, see example below. The interbasin region delineates the area of 
interest within the bouding box such that the same river does not enter the model domain
more than one time, where only the most downstream contiguous area is kept. The delineation
is based on gridded flow direction data and provides a good setup to set upstream
dischage boundary points.

A typical workflow to setup a riverine model schematization is privided in the
:download:`sfincs_riverine.ini <../_examples/sfincs_riverine.ini>` and shown below. 
Each section corresponds to one of the model :ref:`components` and the `[global]` section can be used to pass
aditional arguments to the :py:class:`~hydromt.models.sfincs.SfincsModel`. initialization. 


.. code-block:: console

    hydromt build sfincs /path/to/model_root "{'interbasin': [xmin, ymin, xmax, ymax], buffer: '20'}" -i sfincs_coastal.ini -vv


.. literalinclude:: ../_examples/sfincs_riverine.ini
   :language: Ini


.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options