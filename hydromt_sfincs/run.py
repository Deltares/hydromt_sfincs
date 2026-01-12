"""
SFINCS run function
"""

import os
import subprocess
import sys
from os.path import join
from typing import Union, Optional
from pathlib import Path


def run_sfincs(
    model_root: Union[Path, str],
    sfincs_exe: Optional[Union[Path, str]] = None,
    vm: Optional[str] = None,
    docker_tag: str = "latest",
    verbose: bool = True,
) -> None:
    """
    Run the SFINCS model.

    Either sfincs_exe or vm must be specified.
    If both are specified, sfincs_exe is used.

    Parameters
    ----------
    model_root : Path, str
        The root directory of the model.
    sfincs_exe : Path, str, optional
        The path to the SFINCS executable, by default None.
    vm : str, optional
        The virtual machine to use, either 'docker' or 'singularity', by default None.
    docker_tag : str, optional
        The tag of the docker image to use, by default 'latest'.
        See https://hub.docker.com/r/deltares/sfincs-cpu/tags for available tags.
        Only used if vm is not None.
    verbose : bool, optional
        If True, print the output of the SFINCS model to the console, by default True.
    """
    # check model_root
    if not os.path.isabs(model_root):
        model_root = os.path.abspath(model_root)
    model_root = str(model_root)
    sfincs_inp = join(model_root, "sfincs.inp")
    if not os.path.exists(sfincs_inp):
        raise ValueError(f"SFINCS inp file does not exist: {sfincs_inp}")

    # set command to run depending on OS and VM
    if sfincs_exe is not None and sys.platform == "win32":
        if not os.path.isabs(sfincs_exe):
            sfincs_exe = os.path.abspath(sfincs_exe)
        if not os.path.exists(sfincs_exe):
            raise ValueError(f"sfince_exe path does not exist: {sfincs_exe}")
        cmd = str(sfincs_exe)
    elif vm is not None:
        if vm == "docker":
            cmd = f"docker run -v {model_root}://data deltares/sfincs-cpu:{docker_tag}"
        elif vm == "singularity":
            cmd = f"singularity run -B{model_root}:/data --nv docker://deltares/sfincs-cpu:{docker_tag}"
        else:
            raise NotImplementedError(
                f"{vm} not supported, use 'docker' or 'singularity'"
            )
    else:
        if sys.platform == "win32":
            raise ValueError("sfince_exe must be specified for Windows")
        else:
            raise ValueError("vm must be specified for Linux or macOS")
    if verbose:
        print(f">> {cmd}\n")

    # run & write log file
    with subprocess.Popen(
        cmd,
        cwd=model_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,  # get string output instead of bytes
    ) as proc:
        with open(join(model_root, "sfincs_log.txt"), "w") as f:
            for line in proc.stdout:
                if verbose:
                    print(line.rstrip("\n"))
                f.write(line)
            for line in proc.stderr:
                if verbose:
                    print(line.rstrip("\n"))
                f.write(line)
        proc.wait()
        return_code = proc.returncode

    # check return code
    if vm is not None and return_code == 127:
        raise RuntimeError(
            f"{vm} not found. Make sure it is installed, running and added to PATH."
        )
    elif return_code != 0:
        raise RuntimeError(f"SFINCS run failed with return code {return_code}")

    return None
