.. _sfincs_coastal:

==========================
Setting up a coastal model
==========================

To build a **coastal flood model** with h (waterlevel) and p (precipitation) but no q 
(discharge) boundary conditions, the model region_ is typically defined by a bounding box, 
see example below, or a geometry file.

A typical workflow to setup a coastal model schematization is provided in the
:download:`sfincs_coastal.ini <../_examples/sfincs_coastal.ini>` and shown below. 
Each section corresponds to one of the model :ref:`components` and the `[global]` section can be used to pass
additional arguments to the :py:class:`~hydromt.models.sfincs.SfincsModel`. initialization.
An example is provided in :ref:`examples` section.


.. code-block:: console

    hydromt build sfincs /path/to/model_root "{'bbox': [xmin, ymin, xmax, ymax]}" -i sfincs_coastal.ini -vv


.. literalinclude:: ../_examples/sfincs_coastal.ini
   :language: Ini


.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options


