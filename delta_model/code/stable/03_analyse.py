#%%
from os.path import join
import matplotlib.pyplot as plt

from hydromt_sfincs import SfincsModel, utils
from pathlib import Path

# hmax is computed by SFINCS and read-in from the sfincs_map.nc file
# sfincs_root = "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_1416812" # path to sfincs root

sfincs_root = "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947" # path to sfincs root
# sfincs_root = "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_100RD" # path to sfincs root
# sfincs_root = "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947_100SS" # path to sfincs root

mod = SfincsModel(root = sfincs_root, 
                  data_libs = ['data_catalog_v1.yml'], 
                  mode = "r")

mod.output.read()

# see available output data variables
list(mod.output.data.keys())


#%%
# observation points analyse 
id = [1,2,3,4,5,6,7] # first observation point
# id = [8,9,10,11,12]
mod.output.data['point_zs'][:,id].plot.line(x='time')


#%%
# first we are going to select our highest-resolution elevation dataset
# with the depfile on subgrid resolution this would be:
sfincs_root_dep = "C:/PhD/SFINCS/SFINCS_cloned/output/sfincs_620947"
depfile = join(sfincs_root_dep, "subgrid", "dep_subgrid.tif")

da_dep = mod.data_catalog.get_rasterdataset(depfile)

#%%
# now assuming we have a subgrid model, we don't have hmax available, so we are using zsmax (maximum water levels)
# compute the maximum over all time steps
da_zsmax = mod.output.data["zsmax"].max(dim="timemax")
#%%
# Determine the masking of the floodmap

# Load global dataset and clip to model region
worldcover = mod.data_catalog.get_rasterdataset("esa_worldcover", geom=mod.region, buffer=10)

# Create a mask for water bodies (code 80)
# Reproject to match the high-resolution elevation data
worldcover_reprojected = worldcover.raster.reproject_like(da_dep, method="nearest")
water_mask = worldcover_reprojected == 80


# # Visualize the water mask
# fig, ax = plt.subplots(figsize=(10, 8))
# water_mask.plot(ax=ax)
# plt.tight_layout()
# plt.show()

# print(water_mask)

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

#%%

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


#%%
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
# plt.savefig(join(mod.root, 'figs', 'hmax.png'), dpi=225, bbox_inches="tight")

#%%
# mask water depth
hmin = 0.05
da_h = (mod.output.data["zs"] - mod.output.data["zb"]).copy()
da_h = da_h.where(da_h > hmin).drop("spatial_ref")
da_h.attrs.update(long_name="flood depth", unit="m")


#%%
# create hmax plot and save to mod.root/figs/sfincs_h.mp4
# requires ffmpeg install with "conda install ffmpeg -c conda-forge"
from matplotlib import animation

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
#%%

# Total delta area 
dep_subgrid = mod.data_catalog.get_rasterdataset(
    join(sfincs_root, "subgrid", "dep_subgrid.tif")
)
model_pixels = dep_subgrid.notnull().sum().values
model_area_km2 = (model_pixels * 10 * 10) / 1e6 # convert to km2
print(f"Total model area: {model_area_km2:.2f} km²")

# Total flooded area 
flooded_pixels = da_hmax.notnull().sum().values
total_flooded_area_km2 = (flooded_pixels * (10 * 10)) / 1e6
print(f"Total flooded area: {total_flooded_area_km2:.2f} km²")

# Flood extent (%)
flood_extent_percent = (total_flooded_area_km2 / model_area_km2) * 100
print(f"Flood extent: {flood_extent_percent:.2f} %")


#%%
# Flood depth statistics
flood_depth_mean = da_hmax.mean().values
print(f"Mean flood depth: {flood_depth_mean:.2f} m")

flood_depth_max = da_hmax.max().values
print(f"Max flood depth: {flood_depth_max:.2f} m")

#%%

# Total flood volume (m3 and km3)
total_volume_m3 = (da_hmax * 10 * 10).sum().values
total_volume_km3 = total_volume_m3 / 1e9

print(f"Total flood volume: {total_volume_m3:.0f} m³ = {total_volume_km3:.4f} km³")