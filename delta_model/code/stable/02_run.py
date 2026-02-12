#%%
# Spinup scenario 
from os.path import join
import matplotlib.pyplot as plt
from hydromt_sfincs import SfincsModel, utils
from pathlib import Path
import shutil

root_folder  = Path('C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_spinup') 

# Copy and paste the run.bat file in the output folder and put it into each root folder
shutil.copy2(Path('C:/PhD/SFINCS/SFINCS_cloned/output/run.bat'), root_folder / 'run.bat')
fn = root_folder / "run.bat"
with open(fn, "r") as f:
    txt = f.read()
print(txt)

# Run batch file
import os
cur_dir = os.getcwd()
run_path = str(root_folder)

# Run SFINCS model --------------------------------------------------------------------------
os.chdir(run_path)
os.system("run.bat")
os.chdir(cur_dir)

# open log file to see if everything ran correctly
fn = os.path.join(run_path, "sfincs_log.txt")
with open(fn, "r") as f:
    txt = f.read()
print(txt)

path = str(root_folder) # Inspect what files are in the output folder
dir_list = os.listdir(path)
print(dir_list)


#%%

# RUN OTHER SCENARIOS 
from hydromt._utils import log
import pandas as pd
from hydromt_sfincs import SfincsModel
from datetime import datetime
from pathlib import Path

root_folder = Path('C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_spinup')

sf = SfincsModel(
    data_libs= ["data_catalog_v1.yml"],  # specify which data libraries to use
    root= root_folder,  # specify the root directory for the model
    mode="r"  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite)
)
sf.read()

# change the root and mode to write the updated model to a new location
new_root = Path('C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_100RD')

sf.root.set(new_root, mode="w+")

sf.config.update(
    {
        "rstfile": "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_spinup/sfincs.20181209.000000.rst"
    }
)

# # Initial conditions component - function (water levels along coast) 
# # https://deltares.github.io/hydromt_sfincs/latest/_generated/hydromt_sfincs.components.grid.SfincsInitialConditions.create.html
# sf.initial_conditions.create(
#  ini = x,    # use water levels of spinup 
# )

combined_dataset = pd.read_excel("C:\\PhD\\SFINCS\\SFINCS_cloned\\input\\Combined_dataset_global_deltas.xlsx")
sf.discharge_points.create_timeseries(
    index = [0],
    shape = "gaussian",
    offset = combined_dataset.loc[combined_dataset['BasinID2'] == 620947, 'Discharge_dist'].values[0],
    peak = 700,
    tpeak = 15 * 86400,
    duration = 7 * 86400,
    timestep = 600
)

sf.write()
sf.plot_forcing()

# Run the model 
import shutil

# Copy and paste the run.bat file in the output folder and put it into each root folder
shutil.copy2(Path('C:/PhD/SFINCS/SFINCS_cloned/output/run.bat'), new_root / 'run.bat')

# Check 
fn = new_root / "run.bat"
with open(fn, "r") as f:
    txt = f.read()
print(txt)

# Run batch file and SFINCS ---------------------------------------------------------
import os
cur_dir = os.getcwd()
run_path = str(new_root)

# RUN SFINCS --------------------------------------------------------------------------
os.chdir(run_path)
os.system("run.bat")
os.chdir(cur_dir)

# open log file to see if everything ran correctly
fn = os.path.join(run_path, "sfincs_log.txt")

with open(fn, "r") as f:
    txt = f.read()
print(txt)

# Inspect what files are in the output folder
path = str(new_root)
dir_list = os.listdir(path)
print(dir_list)

#%%
from hydromt._utils import log
import pandas as pd
from hydromt_sfincs import SfincsModel
from datetime import datetime
from pathlib import Path

root_folder = Path('C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_spinup')

sf = SfincsModel(
    data_libs= ["data_catalog_v1.yml"],  # specify which data libraries to use
    root= root_folder,  # specify the root directory for the model
    mode="r"  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite)
)
sf.read()

# change the root and mode to write the updated model to a new location
new_root = Path('C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_100SS')

sf.root.set(new_root, mode="w+")

sf.config.update(
    {
        "rstfile": "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_spinup/sfincs.20181209.000000.rst"
    }
)

# # Initial conditions component - function (water levels along coast) 
# # https://deltares.github.io/hydromt_sfincs/latest/_generated/hydromt_sfincs.components.grid.SfincsInitialConditions.create.html
# sf.initial_conditions.create(
#  ini = x,    # use water levels of spinup 
# )

# Manually create timeseries forcing - using GTSMS for 100-yr RP
sf.water_level.create_timeseries(
    shape = "gaussian", 
    timestep = 600, # 10 minutes in seconds
    offset = 0.5,
    peak = 3, # based on GTSM data previously used 
    tpeak = 15 * 86400,
    duration = 2 * 86400,
)

sf.write()

sf.plot_forcing()

# Run the model 
import shutil

# Copy and paste the run.bat file in the output folder and put it into each root folder
shutil.copy2(Path('C:/PhD/SFINCS/SFINCS_cloned/output/run.bat'), new_root / 'run.bat')

# Check 
fn = new_root / "run.bat"
with open(fn, "r") as f:
    txt = f.read()
print(txt)

# Run batch file and SFINCS ---------------------------------------------------------
import os
cur_dir = os.getcwd()
run_path = str(new_root)

# RUN SFINCS --------------------------------------------------------------------------
os.chdir(run_path)
os.system("run.bat")
os.chdir(cur_dir)

# open log file to see if everything ran correctly
fn = os.path.join(run_path, "sfincs_log.txt")

with open(fn, "r") as f:
    txt = f.read()
print(txt)

# Inspect what files are in the output folder
path = str(new_root)
dir_list = os.listdir(path)
print(dir_list)
