=========================================
HydroMT-SFINCS: SFINCS plugin for HydroMT
=========================================

|pypi| |conda_forge| |docs_latest| |docs_stable| |codecov| |license| |doi| |binder|


What is the HydroMT-SFINCS plugin?
-----------------------------------

HydroMT_ (Hydro Model Tools) is an open-source Python package that facilitates the process of
building and analyzing spatial geoscientific models with a focus on water system models.
It does so by automating the workflow to go from raw data to a complete model instance which
is ready to run and to analyse model results once the simulation has finished.
This plugin provides an implementation of the model API for the SFINCS_ model.

What is SFINCS?
---------------
SFINCS is Deltares' new open-source reduced-complexity model designed for super-fast modelling of compound flooding events in a dynamic way!
What HydroMT-SFINCS does provide is a powerful Python based set of tools to help you build and analyse the best possible SFINCS models!
This HydroMT-SFINCS plugin does not include the SFINCS model or executable itself, for that see the `SFINCS download portal <https://download.deltares.nl/en/download/sfincs/>`_
or the source code repository on `Github <https://github.com/Deltares/SFINCS>`_.
For general documentation about the model, how to run it and what the input files are see the `SFINCS documentation <https://sfincs.readthedocs.io/en/latest/>`_.

Why HydroMT-SFINCS?
-------------------
Setting up hydrodynamic models typically requires many (manual) steps
to process input data and might therefore be time consuming and hard to reproduce.
Especially improving models based on global geospatial datasets, which are
rapidly becoming available at increasingly high resolutions, might be challenging.
HydroMT-SFINCS aims to make the model building process **fast**, **modular** and **reproducible**
and to facilitate the analysis of SFINCS model results

How to use HydroMT-SFINCS?
--------------------------
The HydroMT-SFINCS plugin can be used as a **command line + configuration file** application, which provides commands to *build*,
*update* the SFINCS model with a single line, or **from python** to exploit its rich interface.
You can learn more about how to use HydroMT-SFINCS in its `online documentation. <https://deltares.github.io/hydromt_sfincs/latest/getting_started/intro>`_
For a smooth installing experience we recommend installing HydroMT-SFINCS and its dependencies
from conda-forge in a clean environment, see `installation guide. <https://deltares.github.io/hydromt_sfincs/latest/getting_started/installation>`_

How to cite?
------------
To reference the software please use the the DOI provided in the Zenodo badge that points to the latest release |doi|.

The following paper presents a real-world application of HydroMT-SFINCS:

    Eilander, D., Couasnon, A., Leijnse, T., Ikeuchi, H., Yamazaki, D., Muis, S., Dullaart, J., Haag, A., Winsemius, H. C., and Ward, P. J.:
    A globally applicable framework for compound flood hazard modeling, Nat. Hazards Earth Syst. Sci., 23, 823–846, https://doi.org/10.5194/nhess-23-823-2023, 2023.

How to contribute?
-------------------
If you find any issues in the code or documentation feel free to leave an issue on the `github issue tracker. <https://github.com/Deltares/hydromt_sfincs/issues>`_
You can find information about how to contribute to the HydroMT project at our `contributing page. <https://deltares.github.io/hydromt/latest/dev/contributing>`_

HydroMT seeks active contribution from the (hydro) geoscientific community.
So far, it has been developed and tested with a range of `Deltares <https://www.deltares.nl/en/>`_ models, but
we believe it is applicable to a much wider set of geoscientific models and are
happy to discuss how it can be implemented for your model.

.. _Hydromt: https://deltares.github.io/hydromt/latest/
.. _SFINCS: https://sfincs.readthedocs.io/en/latest/

.. |codecov| image:: https://codecov.io/gh/Deltares/hydromt_sfincs/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt_sfincs

.. |docs_latest| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/hydromt_sfincs/latest
    :alt: Latest developers docs

.. |docs_stable| image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
    :target: https://deltares.github.io/hydromt_sfincs/stable
    :alt: Stable docs last release

.. |pypi| image:: https://badge.fury.io/py/hydromt_sfincs.svg
    :target: https://pypi.org/project/hydromt_sfincs/
    :alt: Latest PyPI version

.. |conda_forge| image:: https://anaconda.org/conda-forge/hydromt_sfincs/badges/version.svg
    :target: https://anaconda.org/conda-forge/hydromt_sfincs

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt_sfincs/main?urlpath=lab/tree/examples

.. |doi| image:: https://zenodo.org/badge/356210788.svg
    :alt: Zenodo
    :target: https://zenodo.org/badge/latestdoi/356210788

.. |license| image:: https://img.shields.io/conda/l/conda-forge/hydromt_sfincs
    :alt: License
    :target: https://github.com/Deltares/hydromt_sfincs/blob/main/LICENSE
