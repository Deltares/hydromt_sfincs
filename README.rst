hydroMT-sfincs: sfincs plugin for hydroMT
###########################################

FIXME add badges with correct links

.. note::

  This minimal branch from the hydromt_sfincs plugin can be used as a **template** to easily 
  implement new plugins for hydroMT. To implement a new model do:
  
  - replace all instances of `plugin` with the model name
  - edit license (default GPLv3)
  - edit model class (plugin.py) to be adapted for the new model
  - check and fix git installation (including pyproject.toml) including entry point
  - test "hydromt --models"
  - edit installation guide 
  - edit .github actions and envs/hydromt-plugin.yml environment
  - edit template documentation


hydroMT_ is a python package, developed by Deltares, to build and analysis hydro models.
It provides a generic model api with attributes to access the model schematization,
(dynamic) forcing data, results and states. This plugin provides an implementation 
for the sfincs model.


.. _hydromt: https://deltares.github.io/hydromt


Installation
------------

FIXME installation guide

Documentation
-------------

Learn more about hydroMT in its `online documentation <https://deltares.github.io/hydromt_sfincs/>`_

Contributing
------------

You can find information about contributing to hydroMT at our `Contributing page <https://deltares.github.io/hydromt_sfincs/latest/contributing.html>`_.

License
-------

Copyright (c) 2021, Deltares

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General 
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your 
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
for more details.

You should have received a copy of the GNU General Public License along with this program. If not, 
see <https://www.gnu.org/licenses/>.
