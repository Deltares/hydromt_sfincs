.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt_sfincs/main?urlpath=lab/tree/examples

This folder contains several iPython notebook examples for **HydroMT-SFINCS**.

These examples can be run online or on your local machine.
To run these examples online press the **binder** badge above.

Local installation
------------------

To run these examples on your local machine you need a copy of the examples folder
of the repository and an installation of HydroMT-SFINCS including some additional
packages required to run the notebooks.

1 - Install HydroMT-SFINCS
**************************

The first step is to install HydroMT-SFINCS and the other Python dependencies in a separate environment,
see the **Install HydroMT-SFINCS in a new environment** section in the
`installation guide <https://deltares.github.io/hydromt_sfincs/latest/getting_started/installation.html>`_


2 - Download the content of the HydroMT github repository
*********************************************************
To run the examples locally, you will need to download the content of the examples folder in the HydroMT-SFINCS repository.
You can  do a `manual download <https://github.com/Deltares/hydromt_sfincs/archive/refs/heads/main.zip>`_
and extract the content of the downloaded ZIP folder **or** clone the repository locally (this requires git):

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt_sfincs.git

3 - Running the examples
************************
Finally, start a jupyter lab server inside the **examples** folder
after activating the **hydromt-sfincs** environment, see below.

Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_.

.. code-block:: console

  $ conda activate hydromt-sfincs
  $ cd hydromt-sfincs/examples
  $ jupyter lab
