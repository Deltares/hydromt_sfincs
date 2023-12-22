.. _sfincs_run:

================================
Running a model
================================

**SFINCS** can be run on multiple different platforms, both local, HPC and cloud based.
The simplest way is to run SFINCS on Windows using a batch-file, which is shown in `this example <../_examples/run_sfincs_model.ipynb>`_.
The example consists of a simple compound flood model that has been created with **HydroMT-SFINCS**.
The model is situated in **Northern Italy** and is forced with waterlevel and discharge boundaries.

If you want to read more about running SFINCS on different platforms, please read the `SFINCS manual <https://sfincs.readthedocs.io/en/latest/example.html#running-sfincs>`_.

.. Note::
    HydroMT-SFINCS does **not** contain the SFINCS kernel itself.
    For more information on how to obtain and install the SFINCS kernel,
    please read the `SFINCS manual (executable) <https://sfincs.readthedocs.io/en/latest/example.html#executable>`_.


.. toctree::
    :hidden:

    Example: Running a model <../_examples/run_sfincs_model.ipynb>
