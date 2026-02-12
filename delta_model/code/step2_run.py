#%%
# Spinup scenario 
import subprocess
import sys
from typing import Union, Optional
from pathlib import Path

def run_sfincs_model(
    model_root: Union[Path, str],
    sfincs_exe: Union[Path, str],
    verbose: bool = True,
) -> None:
    """
    Run the SFINCS model
    """
    # Convert path to absolute to avoid confusion
    model_root = Path(model_root).resolve()
    sfincs_inp = model_root / "sfincs.inp"
    
    if not sfincs_inp.exists():
        raise ValueError(f"SFINCS inp file does not exist: {sfincs_inp}")

    cmd = str(Path(sfincs_exe).resolve())

    if verbose:
        print(f"Running SFINCS in: {model_root}")
        print(f"Using Executable: {cmd}\n")

    # Run the model
    with subprocess.Popen(
        cmd,
        cwd=str(model_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, 
        universal_newlines=True,
    ) as proc:
        log_path = model_root / "sfincs_log.txt"
        with open(log_path, "w") as f:
            for line in proc.stdout:
                if verbose:
                    print(line, end="")
                f.write(line)
        
        proc.wait()
        
    if proc.returncode != 0:
        print(f"\n[!] SFINCS failed with return code {proc.returncode}")
    else:
        print(f"\n[+] SFINCS run completed successfully.")

