.. _sfincs_compound:

==========================
Setting up a compound flood model
==========================

To build a **compound flood model** with waterlevel and upstream river dischage boundary conditions, the model region_ is typically defined by a bounding box, 
see example below, or a geometry file.

A typical workflow to setup a compound flood model schematization is provided in the
:download:`sfincs_simple_compound_base.ini <../_examples/sfincs_simple_compound_base.ini>` and shown below. 
Each section corresponds to :ref:`a model method <model_methods>` and the `[global]` section can be used to pass
additional arguments to the :py:class:`~hydromt.models.sfincs.SfincsModel`. initialization.
An example is provided in :ref:`examples` section.

.. code-block:: console

    hydromt build sfincs /path/to/model_root --region "{'geom': 'data/region.geojson'}" -i sfincs_simple_compound_base.ini -vv

.. literalinclude:: ../_examples/sfincs_simple_compound_base.ini
   :language: Ini

.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
