hydroMT-sfincs: SFINCS plugin for hydroMT
#########################################

.. image:: https://codecov.io/gh/Deltares/hydromt_sfincs/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/Deltares/hydromt_sfincs

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/hydromt_sfincs/latest
    :alt: Latest developers docs

.. image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
    :target: https://deltares.github.io/hydromt_sfincs/stable
    :alt: Stable docs last release

.. image:: https://badge.fury.io/py/hydromt_sfincs.svg
    :target: https://pypi.org/project/hydromt_sfincs/
    :alt: Latest PyPI version

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt_sfincs/main?urlpath=lab/tree/examples


hydroMT_ is a python package, developed by Deltares, to build and analysis hydro models.
It provides a generic model api with attributes to access the model schematization,
(dynamic) forcing data, results and states. This plugin provides an implementation 
for the SFINCS_ model.


.. _hydromt: https://deltares.github.io/hydromt
.. _SFINCS: https://sfincs.readthedocs.io/en/latest/


Installation
------------

hydroMT-sfincs is availble from pypi and we are working on adding a release from conda-forge (ongoing).

To install hydromt_sfincs using pip do:

.. code-block:: console

  pip install hydromt_sfincs

We recommend installing a hydromt-sfincs environment including the hydromt_sfincs package
based on the environment.yml file. This environment will install all package dependencies 
including the core of hydroMT_.

.. code-block:: console

  conda env create -f binder/environment.yml
  conda activate hydromt-sfincs
  pip install hydromt_sfincs

Documentation
-------------

Learn more about the hydroMT_sfincs plugin in its `online documentation <https://deltares.github.io/hydromt_sfincs/>`_

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
