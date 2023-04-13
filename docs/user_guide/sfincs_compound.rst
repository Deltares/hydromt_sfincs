.. _sfincs_example_cli:

==========================
Example: build from CLI
==========================

To build a **compound flood model** with waterlevel and upstream river dischage boundary conditions, the model region_ is typically defined by a bounding box or a geometry file.

A typical workflow to setup a compound flood model schematization is provided in the
:download:`sfincs_base_build.yml <../_examples/sfincs_base_build.yml>` and shown below. 
Each section corresponds to :ref:`a model method <model_methods>` and the `global` section can be used to pass
additional arguments to the :py:class:`~hydromt.models.sfincs.SfincsModel`. initialization.
An example is provided in :ref:`examples` section.

.. code-block:: console

    hydromt build sfincs /path/to/model_root -r "{'geom': 'data/region.geojson'}" -i sfincs_base_build.yml -vv

.. literalinclude:: ../_examples/sfincs_base_build.yml
   :language: yml

.. _region: https://deltares.github.io/hydromt/latest/user_guide/model_region.html
