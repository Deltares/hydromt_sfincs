
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
from hydromt_sfincs.components.forcing import rivers
from sfincs_utils import run_sfincs
from hydromt import DataCatalog
from shapely.geometry import Point
import pandas as pd
import numpy as np
from pyproj import Transformer
import logging
# from river_processing import river_depth_estimation

def build_sfincs_model(delta_basin_id: int, root_folder: Path, data_libs: list = ['data_catalog_v1.yml']):
    
    """
    Build a SFINCS model for a specific basin ID (specified in main.py?)
    
    Parameters
    ----------
    delta_basin_id : int
        The  ID of the delta (dataset specified in data_catalog_v1.yml)
    root_folder : Path
        The root folder where the model will be saved.
    data_libs : list 
        Data_catalog.yml to use 
    """

    # 1a: Configure model - data catalog ------------------------------------------------------------------------------
    catalog = hydromt.DataCatalog(data_libs = data_libs) 

    # 1b: Configure model - define model domain -----------------------------------------------------------------------
    deltas = catalog.get_geodataframe('4_small_deltas')
    delta_domain = deltas[deltas['BasinID2'] == delta_basin_id]

    # 1c: Initialise model -----------------------------------------------------------------------------------------------
    figs_dir = root_folder / "build"
    
    figs_dir.mkdir(parents=True, exist_ok=True)
    def save_fig(fig, name):
        path = figs_dir / f"{name}.png"
        fig.savefig(path, dpi=225, bbox_inches='tight')
        print(f"Saved: {path}")

    log.initialize_logging()
    log.set_log_level(log_level=20)                     # NOTSET=0-9, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50 # added.
    proj = ccrs.PlateCarree()
    plt.ioff()                                          # Turn off interactive plotting 

    log_file = Path(root_folder) / "hydromt_sfincs.log" # Add file handler to log to a file
    log._add_filehandler(log_file)
    logger = logging.getLogger("hydromt")               # Get the logger that HydroMT uses

    sf = SfincsModel(
        data_libs= data_libs,                           # specify which data libraries to use
        root= root_folder,                              # specify the root directory for the model
        mode="w+",                                      # specify the mode for opening the model (r=read only, r+=append, w=write, w+=overwrite
        write_gis=True,                                 # specify whether to write GIS data
    )

    logger.info(f"--- Starting model setup for Basin ID: {delta_basin_id} ---")

    # 1d: Plot model domain -------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(10,7))
    ax = plt.subplot(projection=proj)
    ax.add_image(cimgt.QuadtreeTiles(), 12)

    # Plot the individual basin boundaries 
    delta_domain['geometry'].boundary.plot(ax=ax, label='Individual basins', lw=1, color='yellow', alpha=0.5)

    # Load and plot coastlines near the delta region
    global_water = catalog.get_geodataframe('osm_water', geom=delta_domain)
    global_water.plot(ax=ax, color='red', label='Coastline', alpha = 0.5)
    ax.set_extent(delta_domain['geometry'].total_bounds[[0,2,1,3]], crs=proj)
    ax.legend()
    save_fig(fig, "01_model_domain")

    # 2: Define model grid -----------------------------------------------------------------------------------------
    sf.grid.create_from_region(
        region = {'geom': delta_domain},               # use combined polygon for model domain
        res = 100,                                      # 100m resolution 
        rotated=False,                                  # non-rotated grid. 
        crs='utm'                                       # automatically the closest UTM zone is selected (unit is in meters), 
    )
    fig, ax = sf.plot_basemap(plot_region=True,bmap='sat')
    save_fig(fig, "02_grid")

    # 3: Elevation component setup ---------------------------------------------------------------------------------
    elevation_list = [
        {'elevation': 'merit', 'zmin': 0.001},          # merit looks higher resolution?
        {'elevation': 'gebco'}
    ]
    dep = sf.elevation.create(elevation_list = elevation_list)
    fig, ax = sf.plot_basemap(variable='dep', bmap='sat', plot_region=True)
    save_fig(fig, "03_elevation")


    # 4: Mask component: Define active cells -----------------------------------------------------------------------
    sf.mask.create_active(include_polygon = delta_domain)
    sf.mask.create_boundary(                            # This creates the coastal boundary as active cells 
        btype='waterlevel',
        include_polygon = global_water,                 # Global water works because i've created my own polygons 
        reset_bounds=True
    )
    fig, ax = sf.plot_basemap(
        variable="mask", plot_region=False, plot_bounds=False, bmap="sat", zoomlevel=12
    )
    save_fig(fig, "04_mask")


    # 5: Surface roughness setup -----------------------------------------------------------------------------------
    roughness_list = [
        {
            'lulc': 'esa_worldcover',                   # landuse/landcover dataset
            'reclass_table': 'esa_worldcover_mapping'   # reclassification table
        }
    ]
    sf.roughness.create(roughness_list)
    fig, ax = sf.plot_basemap(variable="manning", plot_bounds=False, bmap="sat", zoomlevel=12)
    save_fig(fig, "05_roughness")


    # 6: Infiltration data setup ------------------------------------------------------------------------------------
    sf.infiltration.create_cn(
        'gcn250', antecedent_moisture = 'avg', reproj_method= 'med' 
    )
    fig, ax = sf.plot_basemap(variable="scs", plot_bounds=False, bmap="sat", zoomlevel=12)
    save_fig(fig, "06_infiltration")


    # 7: Add rivers data --------------------------------------------------------------------------------------------
    # rivers_clipped = catalog.get_geodataframe("4_small_deltas_rivers", geom=delta_domain) # dataset contains rivdph and rivwth
    rivers_sword = catalog.get_geodataframe("global_SWORD_old", geom=delta_domain)   # Variables: width, slope
    rivers_lin = catalog.get_geodataframe("global_rivers_lin", geom=delta_domain)       # Variables: Q2, width_m

    import geopandas as gpd
    # Join 'lin' attributes to the 'sword' geometry
    rivers_clipped = gpd.sjoin_nearest(
        rivers_sword, 
        rivers_lin, 
        max_distance = 1000,  # maybe specify if there is nothing witin 100m then make Q2 0 and use a different way to calculate?       
        how='left'
    )

    # Calculate river width using power-law relationship 
    c = 7.2
    f = 0.50
    rivers_clipped["rivwth"] = (c * (rivers_clipped["Q2"].astype(float) ** f)).astype(float)

    # Keep the better width data from SWORD (variable = 'width') if available, otherwise keep calculated value from power-law relationship 
    rivers_clipped.loc[rivers_clipped["width"].notna(), "rivwth"] = (rivers_clipped["width"])
    
    a = 0.27
    b = 0.30  
    rivers_clipped["rivdph"] = a * (rivers_clipped["Q2"].astype(float) ** b)

    # Replace 'rivdph' values with the minimum value where they are less than the minimum
    min_rivdph = 0 # Note: Making this value higher or lower can affect results
    rivers_clipped["rivdph"] = (np.where(
        rivers_clipped["rivdph"] < min_rivdph, min_rivdph, rivers_clipped["rivdph"]
    )).astype(float)

    # Create rivers list 
    river_list = [
        {
            'centerlines': rivers_clipped              # need rivwth and rivdph variable 
        }
    ]

    # Set up sub-grid --------------------------------------------------------------------------------------------
    sf.subgrid.create(
        elevation_list= elevation_list,
        roughness_list = roughness_list,
        river_list = river_list,
        nr_subgrid_pixels = 4,   # changed from 10 to 4 for initial testing          # 10 for 10m resolution,             
        write_dep_tif=True,                             # save a cloud-optimized geotiff of the subgrid topography
        write_man_tif=True,
        nrmax=5000,                                     # set tile size a bit larger speed up processing (default 2000)
    )

    # 8: Forcing data setup ------------------------------------------------------------------------------------------
    sf.config.update(
        {
            "tref": datetime(2018, 12, 1), 
            "tstart": datetime(2018, 12, 1), 
            "tstop": datetime(2018, 12, 31), 
            "dtrstout": 259200, # tells sfincs to save the simulation after each 3 days = 86400* 3 (can also be trstout, but that'll only be one save)
            "dtmapout": 259200, # output every 3 days 
            "dtmaxout": 259200 # output every 3 days
        }
    )

    # Waterlevel --------
    sf.water_level.create(geodataset = "gtsm_codec_reanalysis", # Sanne's new data 
                          buffer = 25e3) 

    # Rivers -------------
    sf.rivers.create_river_inflow(
        rivers=rivers_clipped,                         
        reverse_river_geom = True,                    
        keep_rivers_geom=True,
        src_type = 'inflow'
    )

    combined_dataset_deltas = catalog.get_dataframe('combined_dataset_deltas') 
    # Create timeseries based on excel. NOTE the index 0 only adds discharge to the first river inflow point, the rest gets zero
    sf.discharge_points.create_timeseries(
        index = [0],
        shape = "constant",
        offset = combined_dataset_deltas.loc[combined_dataset_deltas['BasinID2'] == delta_basin_id, 'Discharge_dist'].values[0],
        timestep = 600,
    )

    # # more complex way to automatically make the time series 
    # # Use final_Q2 as constant river discharge for each inflow point
    # gdf_src = sf.discharge_points.gdf # TODO check if this is a problem 

    # if not gdf_src.empty:
    #     # Map Q2 values from the rivers to the points using a spatial join
    #     gdf_src_mapped = gpd.sjoin_nearest(
    #         gdf_src, 
    #         rivers_clipped[["geometry", "final_Q2"]].to_crs(sf.crs), 
    #         how="left", 
    #         distance_col="dist"
    #     )
    #     # Ensure one-to-one mapping by keeping the first match if multiple are found
    #     gdf_src_mapped = gdf_src_mapped[~gdf_src_mapped.index.duplicated(keep='first')]
        
    #     # Create a constant timeseries for each point
    #     times = [sf.config.get("tstart"), sf.config.get("tstop")]
    #     dis_df = pd.DataFrame(index=times, columns=gdf_src.index)
    #     for idx in gdf_src.index:
    #         q_val = gdf_src_mapped.loc[idx, "final_Q2"]
    #         dis_df[idx] = float(q_val)
            
    #     # Set the forcing using the create method
    #     sf.discharge_points.create(
    #         timeseries=dis_df
    #     )

    # Plot forcing 
    fig, ax = sf.plot_forcing()
    save_fig(fig, "07_forcing")
    logger.info("Forcing added and plot saved.")

    # 9: Observation points setup --------------------------------------------------------------------------------------
    from shapely.geometry import Point
    from shapely.ops import unary_union

    # Merge all segments into one single "LineString" or "MultiLineString"
    # This removes overlaps and treats the whole set as one system
    combined_rivers = unary_union(rivers_clipped.geometry)

    points_list = []
    current_dist = 0
    for i in range(10):
        # Interpolate a point at the calculated distance
        pt = combined_rivers.interpolate(current_dist)
        points_list.append({'obs_id': i + 1, 'geometry': pt})
        current_dist += combined_rivers.length / 9                                # Makes 10 points 
    
    obs_points = gpd.GeoDataFrame(points_list, crs=rivers_clipped.crs)
    sf.observation_points.create(locations=obs_points, merge=False)

    # 10: Plot final model setup ------------------------------------------------------------------------------------- 
    fig, ax = sf.plot_basemap(variable='dep',bmap='sat')
    save_fig(fig, "08_final_model")

    # Save the model ------------------------------------------------------------------------------------------------
    sf.write()
    logger.info(f"Model saved to {root_folder}")


#%%
# To test this script above with an example independently
if __name__ == "__main__":
    delta_basin_id = 2433835 
    root_folder = Path('C:/PhD/SFINCS/SFINCS_cloned/output') / f'sfincs_{delta_basin_id}'
    build_sfincs_model(delta_basin_id, root_folder)


# %%

