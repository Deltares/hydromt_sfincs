# This script merges the hydrobasins that overlap with the delta polygons 

import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
from tqdm import tqdm

def merge_hydrobasins():
    # 1. Gather all your shapefiles
    # Adjust this path if necessary to match your actual data location
    folder = Path("C:/PhD/SFINCS/SFINCS_Kiara/scripts/data")
    
    if not folder.exists():
        print(f"Error: Folder {folder} does not exist.")
        return

    hybas_folders = [f for f in os.listdir(folder) if f.startswith("hybas_")]
    shapefiles = []
    for hybas_folder in hybas_folders:
        path = folder / hybas_folder
        if path.is_dir():
            shapefiles.extend([path / f for f in os.listdir(path) if f.endswith(".shp")])

    if not shapefiles:
        print("No shapefiles found starting with 'hybas_' in the data folder.")
        return

    # 2. Read and combine them
    print(f"Found {len(shapefiles)} shapefiles. Reading and merging...")
    dfs = []
    for f in tqdm(shapefiles):
        try:
            dfs.append(gpd.read_file(f))
        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not dfs:
        print("No dataframes created.")
        return

    full_gdf = pd.concat(dfs, ignore_index=True)

    # 3. Save as a single GeoPackage (this creates the spatial index automatically)
    output_path = folder / "hydrobasins_global.gpkg"
    print(f"Saving to {output_path}...")
    full_gdf.to_file(output_path, driver="GPKG")
    print("Done! You can now use 'hydrobasins_global' in your data catalog.")

if __name__ == "__main__":
    merge_hydrobasins()
