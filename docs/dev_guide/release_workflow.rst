.. _release_workflow:

Release workflow
================

This page describes the lightweight manual release process for HydroMT-SFINCS.
It covers release candidates (RCs) and final releases; it does not use the
automated release-branch workflows from HydroMT core.

The package version is defined by ``__version__`` in
``hydromt_sfincs/__init__.py``. Release tags start with ``v``. For example,
the release branch and tags for version ``2.0.0`` are
``release/v2.0.0``, ``v2.0.0-rc1``, and ``v2.0.0``.

Release candidate
-----------------

1. Merge the changes intended for the release into ``main`` through the usual
   pull-request process. Merge ``main`` into the release branch
   (for example, ``release/v2.0.0``) and run or check the CI tests there.
   Create the release branch from ``main`` if it does not exist yet.
2. Set ``__version__`` in ``hydromt_sfincs/__init__.py`` to the next candidate
   version, such as ``2.0.0-rc4``. Update the top heading in
   ``docs/changelog.rst`` to match and ensure the candidate's changes are
   described.
3. Commit and push the version and changelog updates to the release branch.
   Tag that commit and push the tag:

   .. code-block:: console

      git tag -a v2.0.0-rc4 -m "Release candidate 4 for v2.0.0"
      git push origin release/v2.0.0
      git push origin v2.0.0-rc4

   Use the appropriate version and RC number, and do not reuse an existing
   tag. Pushing a ``v*`` tag starts the publish workflow: it builds and checks
   the package, then uploads it to TestPyPI.
4. To make the candidate available as a GitHub pre-release, create and publish
   a pre-release using that tag. Note that publishing any GitHub release,
   including a pre-release, also triggers the workflow to upload the package
   to PyPI.
5. Test the candidate and make any required fixes on the release branch.
   Merge those fixes into ``main`` as well, or merge the release branch back
   into ``main`` after the final release.

Final release
-------------

1. Once the release candidate is accepted, make sure the release branch
   contains all intended changes and fixes. Run or check the CI tests.
2. Set ``__version__`` to the final version (for example, ``2.0.0``). Update
   the changelog heading with the final version and release date, and finalize
   the release notes.
3. Commit and push these updates to the release branch. Create and push a tag
   on that commit:

   .. code-block:: console

      git tag -a v2.0.0 -m "HydroMT-SFINCS v2.0.0"
      git push origin release/v2.0.0
      git push origin v2.0.0

   Pushing the tag builds and checks the package and uploads it to TestPyPI.
4. Create a GitHub release from the final tag and publish it as a regular
   release, not a pre-release. This triggers the package upload to PyPI.
   Check the Actions run and the published package on PyPI.
5. This repository does not have HydroMT core's automated
   `record-release workflow <https://deltares.github.io/hydromt/latest/dev/core_dev/release.html>`_.
   After the final release, create a pull request from the release branch back
   into ``main`` and merge it with a regular merge so the released version and
   any release-only fixes are preserved. Set ``__version__`` on ``main`` to
   the next development version, using a ``.dev0`` suffix (for example,
   ``2.1.0.dev0``). Add a fresh unreleased section to the changelog with
   ``Added``, ``Changed``, ``Fixed``, ``Deprecated``, and ``Removed``
   subsections.
