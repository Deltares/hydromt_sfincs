#%%
import os
from pathlib import Path
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr 
import hydromt
from hydromt._utils import log
from hydromt_sfincs import SfincsModel
from sfincs_utils import run_sfincs
from hydromt import DataCatalog
from shapely.geometry import Point
import pandas as pd
import numpy as np
from pyproj import Transformer
import logging

main_dir = Path(r"C:\PhD\SFINCS\SFINCS_cloned\input")

# 1a: Configure model - data catalog -----------------------------------------------------
catalog = hydromt.DataCatalog(data_libs = ['data_catalog_v1.yml']) 

# 1b: Configure model - define model domain ----------------------------------------------------
deltas = catalog.get_geodataframe('40_delta_polygons')

# Deltas
target_basin_id = 620947 # small delta in australia (test)
# target_basin_id = 1416812 # zambezi -- slighlty changed geojson to include whole coast 
# target_basin_id = 2433835 # ebro 
# target_basin_id = 2180336 # indus 

region = deltas[deltas['BasinID2'] == target_basin_id]

# Load the hydrobasins that intersect with the selected delta
gdf_basin = catalog.get_geodataframe(
    'hydrobasins_global',
    geom=region # only load basins that intersect with our region of interest
)

# Create a combined polygon from all the basins that overlap with the delta
dissolved_basins = gdf_basin.dissolve()  # Dissolve all overlapping basins
total_polygon = dissolved_basins.overlay(region, how='union').dissolve() # this works for aus but not zambezi 

# Initialise model --------------------------------------------------------
root_folder  = Path('C:/PhD/SFINCS/SFINCS_cloned/output') / f'sfincs_{target_basin_id}_spinup'

# Add figs folder and save helper
figs_dir = root_folder / "build"
figs_dir.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    path = figs_dir / f"{name}.png"
    fig.savefig(path, dpi=225, bbox_inches='tight')
    print(f"Saved: {path}")

log.initialize_logging()
log.set_log_level(log_level=20) # NOTSET=0-9, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50 # added.
proj = ccrs.PlateCarree()
plt.ioff()  # Turn off interactive plotting 

# Add file handler to log to a file
log_file = Path(root_folder) / "hydromt_sfincs.log"
log._add_filehandler(log_file)
logger = logging.getLogger("hydromt") # Get the logger that HydroMT uses

sf = SfincsModel(
    data_libs= ["data_catalog_v1.yml"],         # specify which data libraries to use
    root= root_folder,                          # specify the root directory for the model
    mode="w+",                                  # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite
    write_gis=True,                             # specify whether to write GIS data
)

logger.info(f"--- Starting model setup for Basin ID: {target_basin_id} ---")

# Print all data source names source names in catalogs
sources = list(catalog.sources.keys())
for i, source in enumerate(sources, 1):
    print(f"{i}. {source}")




# # print global_rivers variables 
# rivers_clipped = catalog.get_geodataframe("global_rivers_lin", geom=total_polygon)
# print(rivers_clipped.columns)

# TODO: Perhaps change to polygons ??

# Visualize the combined polygon
fig = plt.figure(figsize=(10,7))
ax = plt.subplot(projection=proj)
ax.add_image(cimgt.QuadtreeTiles(), 12)

# Plot the individual basin boundaries 
gdf_basin['geometry'].boundary.plot(ax=ax, label='Individual basins', lw=1, color='yellow', alpha=0.5)
total_polygon.boundary.plot(ax=ax, label='Final model domain', lw=2, color='red')
region.boundary.plot(ax=ax, color='k', label='Original delta polygon', lw=2, linestyle='--')

# Load and plot coastlines near the delta region
global_water = catalog.get_geodataframe('osm_water', geom=total_polygon)
global_water.plot(ax=ax, color='red', label='Coastline', alpha = 0.5)

# plot the coastal boundary layer 
zambezi_waterlevel_boundary = catalog.get_geodataframe('coastal_boundary_polygons', geom=total_polygon)
zambezi_waterlevel_boundary.plot(ax=ax, color='blue', label='Coastal boundary layer', alpha = 0.5)

ax.set_extent(gdf_basin['geometry'].total_bounds[[0,2,1,3]], crs=proj)
ax.legend()
save_fig(fig, "01_model_domain")





#%%

# 2:Define model grid --------------------------------------------------------

sf.grid.create_from_region(
    # region = {'bbox': total_polygon.to_crs('EPSG:4326').total_bounds},  # use combined polygon for model domain (also works)
    region = {'geom': total_polygon},  # use combined polygon for model domain
    res = 100, # 100m resolution 
    rotated=False, # non-rotated grid. 
    crs='utm' # automatically the closest UTM zone is selected (unit is in meters), 
)

fig, ax = sf.plot_basemap(plot_region=True,bmap='sat')
save_fig(fig, "02_grid")

#%%
# fn = r"C:\Users\lasch\Downloads\CODEC_amax_ERA5_1979_2017_coor_mask_GUM_RPS.nc"
# ds = xr.open_dataset(fn)
# print (ds)

#%%
# fn = r"p:\archivedprojects\11205028-c3s_435\01_data\01_Timeseries\timeseries2\waterlevel\reanalysis_waterlevel_hourly_2018_12_v1.nc"
fn = r"P:/archivedprojects_tmp/11210221-gtsm-reanalysis/GTSM-ERA5-E_dataset/waterlevel/reanalysis_waterlevel_hourly_2018_12_v3.nc"
ds = xr.open_dataset(fn)
print(ds)

# print(sf.crs)
# print(ds.station_x_coordinate.max())

# # # slice in space
# fig, ax = plt.subplots(figsize=(8,6))
# ax.set_xlim(sf.bbox[0], sf.bbox[2])  
# ax.set_ylim(sf.bbox[1], sf.bbox[3])
# global_water.plot(ax=ax, color='red', label='Coastline', alpha = 0.5)
# ds.plot.scatter(x="station_x_coordinate", y="station_y_coordinate", ax=ax)


#%%

elevation_list = [
    # {'elevation': 'deltadtm', 'zmin': 0.001},
    {'elevation': 'merit', 'zmin': 0.001},  # merit looks higher resolution?
    {'elevation': 'gebco'}
]

dep = sf.elevation.create(elevation_list = elevation_list)

# Plot the elevation on top of the model region and satellite image. The variable argument sets which model variable to plot
fig, ax = sf.plot_basemap(variable='dep', bmap='sat', plot_region=True)
save_fig(fig, "03_elevation")

#%%
# 4: MASK component: Define active cells -------------------------------------------------------- 

# Define active cells within the combined basin polygon
sf.mask.create_active(include_polygon = total_polygon)

# Manual
waterlevel_boundary = sf.data_catalog.get_geodataframe("coastal_boundary_polygons") 

# This creates the coastal boundary as active cells 
sf.mask.create_boundary(
    btype='waterlevel',
    # include_polygon = "osm_water",
    # include_polygon_buffer = 10,  # buffer in meters around the coastline
    include_polygon = waterlevel_boundary,
    reset_bounds=True
)

fig, ax = sf.plot_basemap(
    variable="mask", plot_region=False, plot_bounds=False, bmap="sat", zoomlevel=12
)
save_fig(fig, "04_mask")

#%%
# Surface roughness setup --------------------------------------------------------
roughness_list = [
    {
        'lulc': 'esa_worldcover',                 # landuse/landcover dataset
        'reclass_table': 'esa_worldcover_mapping' # reclassification table
    }
]

sf.roughness.create(roughness_list)

fig, ax = sf.plot_basemap(variable="manning", plot_bounds=False, bmap="sat", zoomlevel=12)
save_fig(fig, "05_roughness")


#%%
# Infiltration data setup --------------------------------------------------------
sf.infiltration.create_cn(
    'gcn250', antecedent_moisture = 'avg', reproj_method= 'med' 
)

fig, ax = sf.plot_basemap(variable="scs", plot_bounds=False, bmap="sat", zoomlevel=12)
save_fig(fig, "06_infiltration")


#%%

# Add rivers data ----------------
rivers_clipped = catalog.get_geodataframe("global_rivers_lin", geom=total_polygon)

rivers_list = [
    {
        'centerlines': rivers_clipped # need rivwth and rivdph variable 
    }
]

# Riverine depth estimated using bankfull discharge Q --> Andreadis et al. (2013) --> Eilander paper and global rivers paper --> Leopold and Maddock, 1953
# Calculate bankfull discharge from GloFas at each river segment? (or take from Lin et al. 2019 paper) 
# Calculate depth (h) based on bankfull discharge (Q)
a = 0.27  
b = 0.30
rivers_clipped['rivdph'] = a * (rivers_clipped['Q2']**b)

# Verify the calculation 
print(rivers_clipped[['rivdph','rivwth', 'Q2']])

#%%
# Set up sub-grid ---------------------------------------------------------
sf.subgrid.create(
    elevation_list= elevation_list,
    roughness_list = roughness_list,
    river_list = rivers_list,
    nr_subgrid_pixels = 10,             # 10 for 10m resolution,             # Fill in number of subgrid files
    write_dep_tif=True,                 # save a cloud-optimized geotiff of the subgrid topography
    write_man_tif=True,
    nrmax=5000,                         # set tile size a bit larger speed up processing (default 2000)
)

# make sure to write the updated model to the new model root before running sfincs
sf.write()

# sf.subgrid.data

#%%
fn = r"p:\archivedprojects\11205028-c3s_435\01_data\01_Timeseries\timeseries2\waterlevel\reanalysis_waterlevel_hourly_2018_12_v1.nc"
ds = xr.open_dataset(fn)
print(ds)

# fn = r"C:\Users\lasch\Downloads\CODEC_amax_ERA5_1979_2017_coor_mask_GUM_RPS.nc"
# ds = xr.open_dataset(fn)
# print (ds)

# slice in space
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(sf.bbox[0], sf.bbox[2])  
ax.set_ylim(sf.bbox[1], sf.bbox[3])
global_water.plot(ax=ax, color='red', label='Coastline', alpha = 0.5)
ds.plot.scatter(x="station_x_coordinate", y="station_y_coordinate", ax=ax)


#%%

# 4: Add forcing --------------------------------------------------------

sf.config.update(
    {
        "tref": datetime(2018, 12, 1), 
        "tstart": datetime(2018, 12, 1), 
        "tstop": datetime(2018, 12, 31), 
        "dtrstout": 86400 # tells sfincs to save the simulation after each day (can also be trstout, but that'll only be one save)
    }
)

# set up waterlevel forcing 
sf.water_level.create(geodataset = "gtsm_codec_reanalysis_waterlevel_hourly", 
                      buffer = 25e3) # aus delta

# sf.water_level.data
# sf.plot_forcing()

# Manually create timeseries forcing - using GTSMS for 100-yr RP
# sf.water_level.create_timeseries(
#     shape = "gaussian", 
#     timestep = 600, # 10 minutes in seconds
#     offset = 0.0,
#     peak = 2.1, # based on GTSM data previously used 
#     tpeak = 3600 * 12, # peak at 12 hours
#     duration = 3600 * 36 # total duration of 36 hours (1.5 days)
# )


#%%
# Add rivers and discharge forcing  --------------------------------------------------------

# # Add river inflow points 
# sf.rivers.create_river_inflow(
#     hydrography='merit_hydro',
#     river_upa=100,                   # Minimum upstream area threshold for rivers [km2], by default 10.0
#     river_len=10,                    #  Mimimum river length within the model domain threshhold [m], by default 1 km.
#     keep_rivers_geom=True,
# )

# Add river inflow points 
sf.rivers.create_river_inflow(
    rivers=rivers_clipped,          # was 'global_rivers' before
    reverse_river_geom = True,                    
    keep_rivers_geom=True,
)

combined_dataset_deltas = catalog.get_dataframe('combined_dataset_deltas') 

# Create timeseries based on excel 
# NOTE the index 0 only adds discharge to the first river inflow point, the rest gets zero
sf.discharge_points.create_timeseries(
    index = [0],
    shape = "constant",
    offset = combined_dataset_deltas.loc[combined_dataset_deltas['BasinID2'] == target_basin_id, 'Discharge_dist'].values[0],
    # peak = combined_dataset_deltas.loc[combined_dataset_deltas['BasinID2'] == target_basin_id, 'Discharge99'].values[0],
    # tpeak = 15 * 86400,
    # duration = 2 * 86400,
    timestep = 600,
)

# Plot forcing 
fig, ax = sf.plot_forcing()
save_fig(fig, "07_forcing")
logger.info("Forcing added and plot saved.")

#%%

# Set up observation points 
obs_points_fn = sf.data_catalog.get_geodataframe("aus_obs_points")

# create points
sf.observation_points.create(locations=obs_points_fn, merge=False)


#%%
# Final plot of the model 
fig, ax = sf.plot_basemap(variable='dep',bmap='sat')
save_fig(fig, "08_final_model")

#%%
# Save the model --------------------------------------------------------
sf.write()

# saving model creates a standard folder strcture that can be used to read in the model 
def print_directory_tree(directory):
    directory = str(directory)
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            print(f"{indent}  + {f}")

print_directory_tree(str(root_folder))

