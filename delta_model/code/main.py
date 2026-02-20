#%%
from modulefinder import test
from pathlib import Path
import time
import json
from hydromt import DataCatalog
import yaml
from hydromt_sfincs import SfincsModel, utils

from delta_model.code.step1_build import build_sfincs_model 
from delta_model.code.step2_run import run_sfincs_model


#%%
# # Build one modelL -------------------------------------------------------------------------------------------                 
# # delta_basin_id = 1416812 
# delta_basin_id = 620947
# root_folder = f"C:\\PhD\\SFINCS\\SFINCS_cloned\\output\\sfincs_{delta_basin_id}"
# catalog = "data_catalog_v1.yml"
# sfincs_executable = r"C:\PhD\SFINCS\SFINCS_cloned\hydromt_sfincs\delta_model\software\SFINCS_v2.3.0_mt_Faber_release_exe\sfincs.exe"

# print(f"Building SFINCS model for Basin: {delta_basin_id}...")

# build_sfincs_model(
#     delta_basin_id = delta_basin_id,
#     root_folder = Path(root_folder),
#     data_libs = [catalog]
# )

# print(f"Build complete and saved to: {root_folder}")

# # 2. Run SFINCS (for baseline) ----------------------------------------------------------------------------
# print(f"Running SFINCS baseline model...")

# run_sfincs_model(
#     model_root = Path(root_folder),
#     sfincs_exe = sfincs_executable
# )

#%%
# Loop to build multiple models -----------------------------------------------------------------------------
catalog_file = "data_catalog_v1.yml"

catalog = DataCatalog(data_libs=[catalog_file])
delta_polygons = catalog.get_geodataframe('4_small_deltas')
basin_ids = delta_polygons['BasinID2'].unique().tolist()

sfincs_executable = r"C:\PhD\SFINCS\SFINCS_cloned\hydromt_sfincs\delta_model\software\SFINCS_v2.3.0_mt_Faber_release_exe\sfincs.exe"

#%%
for delta_basin_id in basin_ids:
    root_folder = Path(rf"C:\PhD\SFINCS\SFINCS_cloned\output\sfincs_{delta_basin_id}")
    print(f"Building SFINCS model for Basin: {delta_basin_id}...")

    build_sfincs_model(
        delta_basin_id = delta_basin_id,
        root_folder = root_folder,
        data_libs = [catalog_file]
    )

    print(f"Build complete and saved to: {root_folder}")
    

#%%
# 2. Run SFINCS (for baseline) ----------------------------------------------------------------------------

for delta_basin_id in basin_ids:
    root_folder = Path(rf"C:\PhD\SFINCS\SFINCS_cloned\output\sfincs_{delta_basin_id}")
    print(f"Running SFINCS baseline model for Basin: {delta_basin_id}...")

    run_sfincs_model(
        model_root = root_folder,
        sfincs_exe = sfincs_executable
    )



#%%
delta_basin_id = 2444235
root_folder = Path(rf"C:\PhD\SFINCS\SFINCS_cloned\output\sfincs_{delta_basin_id}")

# 3. Visualise time series from obs points to find restarts file ------------------------------------------
mod = SfincsModel(root = Path(root_folder), 
                  data_libs = [catalog_file], 
                  mode = "r")
mod.output.read()

# See available output data variables
# list(mod.output.data.keys())

# Your existing code
id = [1, 2, 3, 4, 5, 6] 
mod.output.data['point_zs'][:, id].plot.line(x='time')

# Determine restart file from time series 
# Day 9? 

#%%

# 4: Run other senarios by overwriting model config -------------------------------------------------------
from hydromt.readers import read_yaml
from hydromt import DataCatalog

scenarios_yaml = read_yaml(r"C:\PhD\SFINCS\SFINCS_cloned\hydromt_sfincs\delta_model\code\scenarios.yml")
# print(list(scenarios_yaml["scenarios"].keys()))
# print(scenarios_yaml["scenarios"]["river_flood"])
# print(scenarios_yaml["scenarios"]["coastal_flood"])

catalog = DataCatalog(data_libs=["data_catalog_v1.yml"])
combined_dataset_deltas = catalog.get_dataframe('combined_dataset_deltas') 

# Loop through all scenarios defined in the YAML file
for scenario in scenarios_yaml["scenarios"].keys():

    print(f"\n--- Processing Scenarios ---")
    steps = scenarios_yaml["scenarios"][scenario]["steps"]

    # Set offset and peak values
    discharge_offset = combined_dataset_deltas.loc[combined_dataset_deltas['BasinID2'] == delta_basin_id, 'Discharge_dist'].values[0]
    discharge_peak = combined_dataset_deltas.loc[combined_dataset_deltas['BasinID2'] == delta_basin_id, 'Discharge99'].values[0]

    # Update the steps with above values
    for step in steps:
        if "discharge_points.create_timeseries" in step:
            step["discharge_points.create_timeseries"]["offset"] = discharge_offset
            step["discharge_points.create_timeseries"]["peak"] = discharge_peak

    # Read the baseline model
    mod = SfincsModel(root=root_folder, mode="r")
    mod.read()

    # Set the new root for this scenario folder
    new_root = Path(f"C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_{delta_basin_id}_{scenario}")
    mod.root.set(new_root, mode="w+")

    # Apply updates from the YAML steps
    mod.update(steps = steps)

    # Write the updated model files to the new folder
    mod.write()

    # mod.plot_forcing()

    # Run the SFINCS model per scenario
    run_sfincs_model(
        model_root = new_root,
        sfincs_exe = sfincs_executable
    )
    
    print(f"Scenario {scenario} finished and results saved to: {new_root}")



#%%    
# Analyse model --------------------------------------------------------
import numpy as np
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

# scenario =  "coastal_flood" # "river_flood" 


# # sfincs_root = f"C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_{delta_basin_id}" # path to sfincs root
# sfincs_root = f"C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_{delta_basin_id}_{scenario}" # path to sfincs root

# mod = SfincsModel(root = sfincs_root, 
#                   data_libs = ['data_catalog_v1.yml'], 
#                   mode = "r")

# first we are going to select our highest-resolution elevation dataset
# with the depfile on subgrid resolution this would be:
sfincs_root_dep = Path(f"C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_{delta_basin_id}")
depfile = sfincs_root_dep / "subgrid" / "dep_subgrid.tif"

da_dep = mod.data_catalog.get_rasterdataset(depfile)

# now assuming we have a subgrid model, we don't have hmax available, so we are using zsmax (maximum water levels)
# compute the maximum over all time steps
da_zsmax = mod.output.data["zsmax"].max(dim="timemax")
# Determine the masking of the floodmap

# Load global dataset and clip to model region
worldcover = mod.data_catalog.get_rasterdataset("esa_worldcover", geom=mod.region, buffer=10)

# Create a mask for water bodies (code 80)
# Reproject to match the high-resolution elevation data
worldcover_reprojected = worldcover.raster.reproject_like(da_dep, method="nearest")
water_mask = worldcover_reprojected == 80

import xarray as xr
import numpy as np
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape

# 1. Ensure your data is a DataArray and cast to int16 (rasterio prefers integers)
mask_data = water_mask.values.astype(np.int16)

# 2. Extract the affine transform
# This defines how pixels translate to real-world coordinates (lat/lon or meters)
transform = water_mask.rio.transform()

# 3. Vectorize the 1s
# 'shapes' pairs of (polygon_geometry, pixel_value)
shapes = rasterio.features.shapes(mask_data, transform=transform)

# 4. Filter for only the '1' values and convert to shapely objects
polygons = [shape(geom) for geom, value in shapes if value == 0]

# 5. Create a GeoDataFrame
gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=water_mask.rio.crs)

# Optional: Dissolve adjacent polygons into a single multipart feature
gdf_dissolved = gdf.dissolve()

# and again, we can use a threshold to mask minimum flood depth
hmin = 0.05

# Fourthly, we downscale the floodmap
da_hmax = utils.downscale_floodmap(
    zsmax = da_zsmax,
    dep = da_dep,
    hmin = hmin,
    gdf_mask = gdf_dissolved # need to convert to polygons first 
    # floodmap_fn= join(sfincs_root, "floodmap.tif") # uncomment to save to <mod.root>/floodmap.tif
)

# Lastly, we create a basemap plot with hmax on top
fig, ax = mod.plot_basemap(
    fn_out=None,
    figsize=(8, 6),
    variable=da_hmax,
    plot_bounds=False,
    plot_geoms=False,
    bmap="sat",
    zoomlevel=11,
    vmin=0,
    vmax=3.0,
    cbar_kwargs={"shrink": 0.6, "anchor": (0, 0)},
)
ax.set_title(f"SFINCS maximum water depth")

#%%
mod.plot_forcing()

# %%
from matplotlib import animation
import matplotlib.pyplot as plt

# Mask water depth
hmin = 0.05
da_h = (mod.output.data["zs"] - mod.output.data["zb"]).copy()
da_h = da_h.where(da_h > hmin).drop("spatial_ref")
da_h.attrs.update(long_name="flood depth", unit="m")

# Make animation plot 
step = 6  # one frame every <step> dtout
cbar_kwargs = {"shrink": 0.6, "anchor": (0, 0)}

def update_plot(i, da_h, cax_h):
    da_hi = da_h.isel(time=i)
    t = da_hi.time.dt.strftime("%d-%B-%Y %H:%M:%S").item()
    ax.set_title(f"SFINCS water depth {t}")
    cax_h.set_array(da_hi.values.ravel())

fig, ax = mod.plot_basemap(
    fn_out=None,
    variable="",
    bmap="sat",
    plot_bounds=False,
    figsize=(11, 7),
    zoomlevel=12,
)
# added self (their xc/yc didnt work)
cax_h = da_h.isel(time=0).plot(
    ax=ax, 
    vmin=0, 
    vmax=3, 
    cmap=plt.cm.viridis, 
    cbar_kwargs=cbar_kwargs,
    add_labels=False # Prevents xc/yc from overwriting axis labels
)
plt.close()  # to prevent double plot

ani = animation.FuncAnimation(
    fig,
    update_plot,
    frames=np.arange(0, da_h.time.size, step),
    interval=250,  # ms between frames
    fargs=(
        da_h,
        cax_h,
    ),
)

# to show in notebook:
from IPython.display import HTML

HTML(ani.to_html5_video())

# %%
