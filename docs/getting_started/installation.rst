.. _installation_guide:

==================
Installation Guide
==================

Prerequisites
=============
For more information about the prerequisites for an installation of the HydroMT package and related dependencies, please visit the
documentation of `HydroMT core <https://deltares.github.io/hydromt/latest/getting_started/installation.html#installation-guide>`_

If you already have a python & conda installation but do not yet have mamba installed, 
we recommend installing it into your *base* environment using:

.. code-block:: console

  $ conda install mamba -n base -c conda-forge


Installation
============

HydroMT-SFINCS is available from pypi and conda-forge. We recommend installing from conda-forge in a new conda environment.

.. Note::

    In the commands below you can exchange `mamba` for `conda`, see
    `here <https://deltares.github.io/hydromt/latest/getting_started/installation.html#installation-guide>`_ for the difference between both.

Install HydroMT-SFINCS in a new environment (recommended!)
----------------------------------------------------------
You can install HydroMT-SFINCS in a new environment called `hydromt-sfincs` together with a few additional dependencies with:

.. code-block:: console

  $ mamba env create -f https://raw.githubusercontent.com/Deltares/hydromt_sfincs/main/environment.yml

Then, activate the environment (as stated by mamba/conda) to start making use of HydroMT-Wflow:

.. code-block:: console

  conda activate hydromt-sfincs

.. Tip::

    If you already have an environment with this name, either remove it with 
    `conda env remove -n hydromt-sfincs` **or** set a new name for the environment 
    by adding `-n <name>` to the line above. 

Install HydroMT-SFINCS in an existing environment
-------------------------------------------------

To install HydroMT-SFINCS in an existing environment execute the command below 
where you replace `<environment_name>` with the name of the existing environment. 
Note that if some dependencies are not installed from conda-forge but from other 
channels the installation may fail.

.. code-block:: console

   $ mamba install -c conda-forge hydrom_sfincs -n <environment_name>

Developer install
==================
To be able to test and develop the HydroMT-SFINCS package see instructions in the :ref:`Developer installation guide <dev_env>`.
