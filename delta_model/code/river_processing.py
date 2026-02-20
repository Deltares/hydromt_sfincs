#%%
# Processing rivers 
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point

# For every basinid in basin_IDs (from the delta polygon dataset in the data catalog):
# apply this function below 

# 1. Load data
def river_depth_estimation(delta_domain, rivers_sword, rivers_lin):
    # xxx

# %%


if __name__ == "__main__":
    # basin_id = 620947
    basin_id = 4267691
    delta_domain_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\4_delta_polygons.geojson"
    rivers_sword_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\SWORD_global_unpublished.gpkg"
    rivers_lin_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\rivers_lin.gpkg"
    output_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers_with_dph_function.gpkg"

    delta_domain = gpd.read_file(delta_domain_path)
    delta_domain = delta_domain[delta_domain['BasinID2'] == basin_id]

    rivers_sword = gpd.read_file(rivers_sword_path, mask=delta_domain)
    rivers_lin = gpd.read_file(rivers_lin_path, mask=delta_domain)

    rivers_clipped = river_depth_estimation(
        delta_domain=delta_domain,
        rivers_sword=rivers_sword,
        rivers_lin=rivers_lin,
    )

    print(rivers_clipped)
    # export as gpkg
    rivers_clipped.to_file(output_path, driver='GPKG')




# %%
