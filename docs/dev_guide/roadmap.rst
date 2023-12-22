.. _roadmap:

Roadmap
=======

Here the roadmap for the HydroMT SFINCS plugin will be presented.

Short-term plans
----------------
- Add the possibility to setup, read and write the pump and culvert to the :py:class:`~hydromt_sfincs.SfincsModel` class. Note that SFINCS already supports this functionality.
- Next to the `inifile`, also support reading and writing of the `rstfile` option to restart a simulation from a previous state.
- Add more eleborate options to account for spatially varying infiltration.
- Add functionality to burn rivers in subgrid tabels.
- Add functionality to locally adjust subgrid tables, e.g. to account for green infrastructure.
- Add support for other forcings such as wind and waves.

Long-term plans
---------------
- Add quadtree gridding functionality to the :py:class:`~hydromt_sfincs.SfincsModel` class. This allows to locally refine the grid.
- Allow for easy coupling to SnapWave to account for wave effects.
