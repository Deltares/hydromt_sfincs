.. _release_workflow:

Release workflow
================

This document will go through the steps of releasing Hydromt-Wflow to Pypi and conda.

1. Create a release branch
""""""""""""""""""""""""""
Create a release branch locally with an appropriate name.

2. Bump the version number
""""""""""""""""""""""""""
Change the version number in the __init__.py file of the hydromt_wflow module.

3. Update the changelog
"""""""""""""""""""""""
Update the changelog header with the version of the release and include the date of release in brackets.
Also add extra information on the highlights of the release.

4. Check the dependencies
"""""""""""""""""""""""""
Update or newly install the default pixi environment and run the tests to check if everything is working with the latest versions of the dependencies.

5. Merge the release branch into main
"""""""""""""""""""""""""""""""""""""
Make sure that everything is included that needs to be released and then merge into main.

6. Create a tag
"""""""""""""""
Check out the main branch and create a tag with the following command.

.. code-block:: console

    $ git tag <version>

Then push the tag to github.

.. code-block:: console

    $ git push origin <version>

This will kickstart the test release workflow to Pypi.

7. Create a release on GitHub and release to Pypi
"""""""""""""""""""""""""""""""""""""""""""""""""
If the test release workflow passes it is time to draft a release on GitHub. Name the release after the new version and choose the corresponding tag.
Then describe the release, you can use the same description from the changelog. Then choose whether the release is a pre-release or the latest version.
Publishing the release will start the release to Pypi.

8. Check the Pypi release (optional)
""""""""""""""""""""""""""""""""""""
Install the latest release with pip to check whether the release is working as intended.

9. Release on conda
-------------------
TODO


10. Create post-release pull request
------------------------------------
Change the __version__ to <version>.dev and add an unreleased header and subheaders with added, changed, fixed, deprecated, and removed to the changelog.
