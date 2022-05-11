.. _installation_guide:

==================
Installation Guide
==================

Prerequisites
=============
For more information about the prerequisites for an installation of the HydroMT SFINCS package please visit the
documentation of the `HydroMT core <https://deltares.github.io/hydromt/preview/getting_started/installation.html#installation-guide>`_

Installation
============

HydroMT SFINCS is available from pypi and conda-forge, but we recommend installing from conda-forge in a new conda environment.

.. Note::

    In the commands below you can exchange `mamba` for `conda`, see
    `here <https://deltares.github.io/hydromt/preview/getting_started/installation.html#installation-guide>`_ for the difference between both.

Install HydroMT in a new environment
------------------------------------
.. Tip::

    This is our recommended way of installing HydroMT SFINCS!

To install HydroMT SFINCS in a new environment called `hydromt-sfincs` from the conda-forge channel do:

.. code-block:: console

  $ conda create -n hydromt-sfincs -c conda-forge hydromt_sfincs

Then, activate the environment (as stated by mamba/conda) to start making use of HydroMT SFINCS:

.. code-block:: console

  conda activate hydromt-sfincs

Besides the SFINCS model plugin, this will install all dependencies including the core HydroMT library.

**Alternatively** to install hydromt_sfincs using pip do (not recommended):

.. Note::

    Make sure this is installed in the same environment as HydroMT.

.. code-block:: console

  $ pip install hydromt_sfincs

Install HydroMT in an existing environment
------------------------------------------

To install HydroMT SFINCS **using mamba or conda** execute the command below after activating the correct environment.
Note that if some dependencies are not installed from conda-forge the installation may fail.

.. code-block:: console

   $ conda install -c conda-forge hydromt_sfincs

For **Using pip** from pypi (not recommended) see above

Developer install
==================
To be able to test and develop the HydroMT package see instructions in the :ref:`Developer installation guide <dev_env>`.
