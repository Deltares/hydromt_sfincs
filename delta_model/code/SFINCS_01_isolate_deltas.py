# This script identifies deltas with significant urban areas based on land use data.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rio
from scipy.ndimage import label
from tqdm import tqdm
import warnings

def main():
    # Define paths
    folder = Path("C:\PhD\SFINCS\SFINCS_Kiara\hands_on\data")
    landuse_path = folder / "Copernicus_LandUse.tif"
    deltas_path = folder / "delta_polygons/769_delta_polygons.geojson"
    
    # Check if files exist
    if not landuse_path.exists():
        print(f"Error: Landuse file not found at {landuse_path}")
        return
    if not deltas_path.exists():
        print(f"Error: Delta polygons file not found at {deltas_path}")
        return

    # Load data
    print("Loading data...")
    try:
        landuse_tiff = rio.open_rasterio(landuse_path)
        delta_polygons = gpd.read_file(deltas_path)
        combined_dataset_path = folder / "Combined_dataset_global_deltas.xlsx"
        combined_dataset = pd.read_excel(combined_dataset_path)

        # Filter for deltas with geomorphic_area > 100
        if 'Geomorphic_Area' in combined_dataset.columns:
            valid_deltas = set(combined_dataset[combined_dataset['Geomorphic_Area'] > 100]['BasinID2'])
            print(f"Filtered to {len(valid_deltas)} deltas with Geomorphic_Area > 100")
        else:
            print("Warning: 'Geomorphic_Area' column not found. No area filtering applied.")
            valid_deltas = set(combined_dataset['BasinID2'])

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure CRS matches
    if delta_polygons.crs != landuse_tiff.rio.crs:
        print("Reprojecting delta polygons to match landuse CRS...")
        delta_polygons = delta_polygons.to_crs(landuse_tiff.rio.crs)

    # Manual list of BasinID2s to include regardless of criteria
    manual_include_ids = [
        1416812, # mozam
        1210764, # south america
        540974, # amazon
        268554,
        2528,
        1258, 
        3648, 
        1873515,
        2433735, 
        2098785,
        2180336,
        866167,
        1086772,
        5491396,
        472622,
        7126506,
        1475442,
        689044,
        1668507, # NZ
        6654226, # jakarta
        689044, # guayaquil
        3355105, # tigris 
        620947, # calvert (small in Aus - Justin)
        1443892 # madagascar east coast 
    ]

    # Manual list of BasinID2s to exclude regardless of criteria
    manual_exclude_ids = [
        619774,
        3928736,
        4553386,
        6928566,
        5355736, 
        2791426,
        2545836,
        2244156,
        6154157,
        1440556,
        1560976,
        1383076,
        6654226,
        2996486,
        3279946,
        6154156,
        2267016
    ]

    urban_deltas = []
    
    print("Processing deltas...")
    # Iterate through each delta
    for index, row in tqdm(delta_polygons.iterrows(), total=len(delta_polygons)):
        basin_id = row['BasinID2']
        
        # Check if manually excluded
        if basin_id in manual_exclude_ids:
            continue

        # Check if manually included
        if basin_id in manual_include_ids:
            urban_deltas.append(basin_id)
            continue

        if basin_id not in valid_deltas:
            continue
        
        try:
            # Clip the land use raster to the polygon
            clipped_landuse = landuse_tiff.rio.clip([row.geometry], from_disk=True)
            
            # Get urban mask (value 50)
            # clipped_landuse is an xarray DataArray. .values gives numpy array.
            # It might have shape (1, height, width), so we select the first band.
            data = clipped_landuse.values[0]
            
            urban_mask = (data == 50).astype(int)
            
            if np.sum(urban_mask) == 0:
                continue

            # Find connected components (clusters)
            # Use 4-connectivity (pixels must share an edge to be considered "together")
            structure = np.array([[0,1,0],
                                  [1,1,1],
                                  [0,1,0]], dtype=int)
            labeled_array, num_features = label(urban_mask, structure=structure)
            
            # Check size of each cluster
            # bincount gives number of pixels for each label. Label 0 is background.
            sizes = np.bincount(labeled_array.ravel())
            
            # We ignore label 0 (background)
            if len(sizes) > 1:
                max_cluster_size = sizes[1:].max()
                # User requested 20km2. Resolution is 100m (0.01km2 per pixel).
                # 20 km2 / 0.01 km2 = 2000 pixels.
                if max_cluster_size >= 2000:
                    urban_deltas.append(basin_id)
                    
        except Exception as e:
            # Clipping might fail if the polygon is outside the raster bounds or other issues
            # print(f"Skipping BasinID {basin_id}: {e}")
            pass

    print(f"\nFound {len(urban_deltas)} deltas with urban clusters >= 20km2 (2000 cells).")
    
    output_geojson = folder / "delta_polygons/40_delta_polygons.geojson"
    # Filter the original geodataframe
    urban_deltas_gdf = delta_polygons[delta_polygons['BasinID2'].isin(urban_deltas)]

    # Add delta_name from combined_dataset
    # Check for likely column names for delta name
    name_col = None
    for col in ['delta_name']:
        if col in combined_dataset.columns:
            name_col = col
            break
    
    if name_col:
        print(f"Adding '{name_col}' to the output GeoJSON...")
        urban_deltas_gdf = urban_deltas_gdf.merge(combined_dataset[['BasinID2', name_col]], on='BasinID2', how='left')
        if name_col != 'delta_name':
            urban_deltas_gdf = urban_deltas_gdf.rename(columns={name_col: 'delta_name'})
    else:
        print("Warning: Could not find a delta name column in combined_dataset.")

    urban_deltas_gdf.to_file(output_geojson)
    print(f"Geojson saved to {output_geojson}")

if __name__ == "__main__":
    main()


# %%
