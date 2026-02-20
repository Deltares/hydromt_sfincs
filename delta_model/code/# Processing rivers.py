# #%%

# # ASSUMES CONSTANT DEPTH DOWN WHOLE RIVER AFTER CONFLUENT OR BIFURCATION POINT 

# # Processing rivers 
# import geopandas as gpd
# import pandas as pd
# import numpy as np
# from shapely.geometry import Point

# # 1. Load data
# delta_domain = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\4_delta_polygons.geojson")
# delta_domain = delta_domain[delta_domain['BasinID2'] == 4267691].to_crs(epsg=3857)

# # delta_domain = gpd.read_file(r"C:\Users\lasch\Downloads\NL_polygon.gpkg").to_crs(epsg=3857)

# rivers_sword = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\SWORD_global_unpublished.gpkg", mask=delta_domain).to_crs(epsg=3857)
# rivers_lin = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\rivers_lin.gpkg", mask=delta_domain).to_crs(epsg=3857)
# rivers_sword_old = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers.gpkg", mask=delta_domain).to_crs(epsg=3857)

# # 2. Map boundary inflows
# boundary_line = delta_domain.geometry.boundary.iloc[0]
# inflow_reaches = rivers_sword[rivers_sword.intersects(boundary_line)].copy()
# joined_inflows = gpd.sjoin_nearest(inflow_reaches, rivers_lin, max_distance=100, how='inner')
# rivers_sword['inflow_Q2'] = rivers_sword['reach_id'].map(dict(zip(joined_inflows['reach_id'], joined_inflows['Q2']))).fillna(0)

# # 3. Create Connection Map
# tail_circles = rivers_sword.copy()
# tail_circles['geometry'] = tail_circles.geometry.apply(lambda x: Point(x.coords[-1])).buffer(150)
# head_points = rivers_sword.copy()
# head_points['geometry'] = head_points.geometry.apply(lambda x: Point(x.coords[0]))

# # Spatial join to find connections
# connections = gpd.sjoin(head_points[['reach_id', 'geometry']], tail_circles[['reach_id', 'geometry']], how='inner', predicate='within')
# # Drop geometry and convert to simple DataFrame for the connection map
# conn_map = pd.DataFrame(connections[['reach_id_left', 'reach_id_right']])
# conn_map = conn_map[conn_map['reach_id_left'] != conn_map['reach_id_right']].astype(int)
# conn_map = conn_map.rename(columns={'reach_id_left': 'down_id', 'reach_id_right': 'up_id'})

# # 4. Flow Accumulation (Waterfall)
# q_dict = rivers_sword.set_index('reach_id')['inflow_Q2'].to_dict()
# is_boundary = rivers_sword.set_index('reach_id')['inflow_Q2'] > 0
# widths = rivers_sword.set_index('reach_id')['width'].to_dict()

# for _ in range(100):
#     next_q = {rid: val for rid, val in q_dict.items() if is_boundary.get(rid, False)}
#     for up_id, group in conn_map.groupby('up_id'):
#         parent_flow = q_dict.get(up_id, 0)
#         if parent_flow <= 0: continue
        
#         down_ids = group['down_id'].unique()
#         total_w = sum(widths.get(d_id, 0) for d_id in down_ids)
#         for d_id in down_ids:
#             if not is_boundary.get(d_id, False):
#                 share = widths.get(d_id, 0) / total_w if total_w > 0 else (1.0 / len(down_ids))
#                 next_q[d_id] = next_q.get(d_id, 0) + (parent_flow * share)
#     q_dict.update(next_q)

# # rename width variable to match sfincs requirements
# rivers_sword = rivers_sword.rename(columns={'width': 'rivwth'})

# # Calculate depth of river segments 
# rivers_sword['final_Q2'] = rivers_sword['reach_id'].map(q_dict).fillna(0)
# rivers_clipped = rivers_sword[rivers_sword['final_Q2'] > 1e-4].copy()

# a = 0.27
# b = 0.30
# rivers_clipped['rivdph'] = (a * (rivers_clipped['final_Q2']**b)).clip(lower=0.1)

# print(rivers_clipped[['inflow_Q2', 'reach_id', 'final_Q2', 'rivwth', 'rivdph']])

# # export as gpkg
# rivers_clipped.to_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\global_SWORD_with_dph.gpkg", driver='GPKG')

#%%

# Processing rivers - FIXED BIFURCATION WITH WIDTH-BASED Q PARTITIONING

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

# 1. Load data
delta_domain = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\4_delta_polygons.geojson")
delta_domain = delta_domain[delta_domain['BasinID2'] == 4267691].to_crs(epsg=3857)
# delta_domain = gpd.read_file(r"C:\Users\lasch\Downloads\NL_polygon.gpkg").to_crs(epsg=3857)
rivers_sword = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\SWORD_global_unpublished.gpkg", mask=delta_domain).to_crs(epsg=3857)
rivers_lin = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\rivers_lin.gpkg", mask=delta_domain).to_crs(epsg=3857)

# 2. Map boundary inflows
boundary_line = delta_domain.geometry.boundary.iloc[0]
inflow_reaches = rivers_sword[rivers_sword.intersects(boundary_line)].copy()
joined_inflows = gpd.sjoin_nearest(inflow_reaches, rivers_lin, max_distance=100, how='inner')
rivers_sword['inflow_Q2'] = rivers_sword['reach_id'].map(dict(zip(joined_inflows['reach_id'], joined_inflows['Q2']))).fillna(0)

# 3. Create Connection Map
tail_circles = rivers_sword.copy()
# takes the last coordinate of every river line and creates a buffer circle 
tail_circles['geometry'] = tail_circles.geometry.apply(lambda x: Point(x.coords[-1])).buffer(150)
head_points = rivers_sword.copy()
# takes the first coordinate of every river line and creates a point geometry (no buffer needed for heads)
head_points['geometry'] = head_points.geometry.apply(lambda x: Point(x.coords[0]))
# finds which heads fall within which tail buffers
connections = gpd.sjoin(head_points[['reach_id', 'geometry']], tail_circles[['reach_id', 'geometry']], how='inner', predicate='within')
# cleans up the connections and makes a clear upstream-downstream mapping 
conn_map = pd.DataFrame(connections[['reach_id_left', 'reach_id_right']])
conn_map = conn_map[conn_map['reach_id_left'] != conn_map['reach_id_right']].astype(int)
conn_map = conn_map.rename(columns={'reach_id_left': 'down_id', 'reach_id_right': 'up_id'})

# 4. distributes water through branches --> FIXED Flow Accumulation - DYNAMIC WIDTHS EACH ITERATION
q_dict = rivers_sword.set_index('reach_id')['inflow_Q2'].to_dict() # dictionary of all river discharges 
is_boundary = rivers_sword.set_index('reach_id')['inflow_Q2'] > 0 # marks where the "faucets" are 

for iteration in range(100):
    # **CRITICAL FIX: Fresh width lookup EVERY iteration**
    current_widths = pd.to_numeric(rivers_sword['width'], errors='coerce').fillna(1.0)
    widths_dict = rivers_sword.set_index('reach_id')['width'].to_dict()
    
    # Temporary staging area for next iteration. Copies only "faucets. Everything else is 0 waiting to receive water from upstream. This ensures we don't overwrite flows before they have been distributed.
    next_q = {rid: val for rid, val in q_dict.items() if is_boundary.get(rid, False)}
    
    # For loop acts as digital pulse pushing the water downstream 
    for up_id, group in conn_map.groupby('up_id'): # for every up_id (upstream river segment), who is waiting downstream to receive? group_by treats every segment that has "children" as a source
        parent_flow = q_dict.get(up_id, 0) # checks how much water is in the upstream segment 
        if parent_flow <= 0: # this checks if there is actually water to move (erach time the iteration is run, water fills up and more reaches will pass this check)
            continue
        
        down_ids = group['down_id'].unique() # finds all the downstream segments (children) that should receive this water
        total_w = sum(widths_dict.get(d_id, 1.0) for d_id in down_ids) # sums the width when theres a bifurcation to know how to split the water (wider rivers get more water, narrower rivers get less water)
        
        # gives water to the "children"
        for d_id in down_ids: # runs twice if there are 2 bifurcation rivers 
            if not is_boundary.get(d_id, False): # to avoid overwriting the inflow from the "faucets"
                w_share = widths_dict.get(d_id, 1.0) 
                share = w_share / total_w if total_w > 0 else (1.0 / len(down_ids)) # fair share calculation (width/total). If width is 0, then it divides 50/50 
                # the maths --> handles bifurcations and confluences at once 
                next_q[d_id] = next_q.get(d_id, 0) + (parent_flow * share) # checks the river to see if another river has already "poured"water into it 
    
    q_dict.update(next_q)

# 5. Apply results and calculate depths
rivers_sword['final_Q2'] = rivers_sword['reach_id'].map(q_dict).fillna(0) # assigns calculated discharge back to dataframe
rivers_clipped = rivers_sword[rivers_sword['final_Q2'] > 1e-4].copy()

# 6. **HYDRAULIC GEOMETRY: Widths CONTROL depths** 
rivers_clipped['rivwth'] = pd.to_numeric(rivers_clipped['width'], errors='coerce').clip(lower=1.0) # ensure width is numeric and at least 1m

# Area = d * w 
# Area = (aQ^b) * (cQ^f) using constants and discharge
# Area = (a * c) * (Q^c+f)
# Area = alpha * Q^beta where alpha = a*c and beta = b+f
a = 0.27
b = 0.3
c = 7.2
f = 0.5
alpha = (a * c)
beta = (b + f)

# alpha, beta = 1.944, 0.98  
q_nonneg = rivers_clipped['final_Q2'].clip(lower=0.0)
rivers_clipped['crosssection_area'] = alpha * (q_nonneg ** beta)

# d = Area / w
rivers_clipped['rivdph'] = rivers_clipped['crosssection_area'] / rivers_clipped['rivwth']
rivers_clipped['rivdph'] = rivers_clipped['rivdph'].clip(lower=0.1, upper=25.0)

# 7. DEBUG & VERIFICATION
print("Bifurcation & Width-Depth check (top 20 reaches):")
verify = rivers_clipped.nlargest(20, 'final_Q2')[['reach_id', 'width', 'rivwth', 'final_Q2', 'rivdph', 'crosssection_area']]
verify = verify.sort_values(['width', 'final_Q2'])
print(verify)
print(f"\nWidth-Q correlation: {rivers_clipped['rivwth'].corr(rivers_clipped['final_Q2']):.3f}")
print(f"Width-Depth correlation: {rivers_clipped['rivwth'].corr(rivers_clipped['rivdph']):.3f} (negative = working!)")

print("\nSFINCS-ready: rivers_clipped with rivwth, rivdph, final_Q2")
print(f"Total reaches: {len(rivers_clipped)}")

# export as gkpg
# rivers_clipped.to_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\global_SWORD_with_dph_wdth_test.gpkg", driver='GPKG')

# EXTRA TEXT
# # 3. Calculate Predicted Width (Theoretical) vs. Observed Width (SWORD)
# q_nonneg = rivers_clipped['final_Q2'].clip(lower=0.0)
# rivers_clipped['predicted_w'] = c * (q_nonneg ** f)

# # 4. Calculate Area and final Depth
# rivers_clipped['required_area'] = alpha * (q_nonneg ** beta)
# rivers_clipped['rivdph'] = rivers_clipped['required_area'] / rivers_clipped['rivwth']

# # 5. NEW: Calculate the "Width Ratio" 
# # (How much wider/narrower is the real river than the theory?)
# rivers_clipped['w_ratio'] = rivers_clipped['rivwth'] / rivers_clipped['predicted_w']

# verify = rivers_clipped.nlargest(20, 'final_Q2')[['reach_id', 'final_Q2', 'rivwth', 'predicted_w', 'w_ratio', 'rivdph']]
# print(verify)


# %%

# FOR LOOP for deltas in data_catalog 
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from hydromt import DataCatalog

# 1. Load data from catalog and loop over deltas
catalog_file = Path(__file__).with_name("data_catalog_v1.yml")
catalog = DataCatalog(data_libs=[str(catalog_file)])

delta_domains = catalog.get_geodataframe("4_small_deltas")[["BasinID2", "geometry"]].to_crs(epsg=3857)
rivers_sword_all = catalog.get_geodataframe("global_SWORD_network").to_crs(epsg=3857)
rivers_lin_all = catalog.get_geodataframe("global_rivers_lin").to_crs(epsg=3857)

# Hydraulic geometry constants
a = 0.27
b = 0.3
c = 7.2
f = 0.5
alpha = (a * c)
beta = (b + f)

per_delta_results = []

for basin_id, delta_domain in delta_domains.groupby("BasinID2"):
    delta_geom = delta_domain.unary_union
    rivers_sword = rivers_sword_all[rivers_sword_all.intersects(delta_geom)].copy()
    rivers_lin = rivers_lin_all[rivers_lin_all.intersects(delta_geom)].copy()

    if rivers_sword.empty:
        print(f"Basin {basin_id}: skipped (no SWORD reaches in domain)")
        continue

    # 2. Map boundary inflows
    boundary_line = delta_geom.boundary
    inflow_reaches = rivers_sword[rivers_sword.intersects(boundary_line)].copy()

    if inflow_reaches.empty or rivers_lin.empty:
        rivers_sword["inflow_Q2"] = 0.0
    else:
        joined_inflows = gpd.sjoin_nearest(inflow_reaches, rivers_lin, max_distance=100, how="inner")
        rivers_sword["inflow_Q2"] = rivers_sword["reach_id"].map(
            dict(zip(joined_inflows["reach_id"], joined_inflows["Q2"]))
        ).fillna(0)

    # 3. Create Connection Map
    tail_circles = rivers_sword.copy()
    tail_circles["geometry"] = tail_circles.geometry.apply(lambda x: Point(x.coords[-1])).buffer(150)
    head_points = rivers_sword.copy()
    head_points["geometry"] = head_points.geometry.apply(lambda x: Point(x.coords[0]))

    connections = gpd.sjoin(
        head_points[["reach_id", "geometry"]],
        tail_circles[["reach_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    conn_map = pd.DataFrame(connections[["reach_id_left", "reach_id_right"]])
    conn_map = conn_map[conn_map["reach_id_left"] != conn_map["reach_id_right"]].astype(int)
    conn_map = conn_map.rename(columns={"reach_id_left": "down_id", "reach_id_right": "up_id"})

    # 4. Flow accumulation with width-based partitioning
    width_numeric = pd.to_numeric(rivers_sword["width"], errors="coerce").fillna(1.0).clip(lower=1.0)
    rivers_sword["width"] = width_numeric
    q_dict = rivers_sword.set_index("reach_id")["inflow_Q2"].to_dict()
    is_boundary = (rivers_sword.set_index("reach_id")["inflow_Q2"] > 0).to_dict()
    widths_dict = rivers_sword.set_index("reach_id")["width"].to_dict()

    for _ in range(100):
        next_q = {rid: val for rid, val in q_dict.items() if is_boundary.get(rid, False)}
        for up_id, group in conn_map.groupby("up_id"):
            parent_flow = q_dict.get(up_id, 0)
            if parent_flow <= 0:
                continue

            down_ids = group["down_id"].unique()
            total_w = sum(widths_dict.get(d_id, 1.0) for d_id in down_ids)

            for d_id in down_ids:
                if not is_boundary.get(d_id, False):
                    w_share = widths_dict.get(d_id, 1.0)
                    share = w_share / total_w if total_w > 0 else (1.0 / len(down_ids))
                    next_q[d_id] = next_q.get(d_id, 0) + (parent_flow * share)

        q_dict.update(next_q)

    # 5. Apply results and calculate depths
    rivers_sword["final_Q2"] = rivers_sword["reach_id"].map(q_dict).fillna(0)
    rivers_clipped = rivers_sword[rivers_sword["final_Q2"] > 1e-4].copy()

    if rivers_clipped.empty:
        print(f"Basin {basin_id}: no routed reaches above threshold")
        continue

    rivers_clipped["rivwth"] = pd.to_numeric(rivers_clipped["width"], errors="coerce").clip(lower=1.0)
    q_nonneg = rivers_clipped["final_Q2"].clip(lower=0.0)
    rivers_clipped["crosssection_area"] = alpha * (q_nonneg**beta)
    rivers_clipped["rivdph"] = (rivers_clipped["crosssection_area"] / rivers_clipped["rivwth"]).clip(lower=0.1, upper=25.0)
    rivers_clipped["BasinID2"] = basin_id

    per_delta_results.append(rivers_clipped)

    print(
        f"Basin {basin_id}: reaches={len(rivers_clipped)}, "
        f"Q-Width corr={rivers_clipped['rivwth'].corr(rivers_clipped['final_Q2']):.3f}, "
        f"Width-Depth corr={rivers_clipped['rivwth'].corr(rivers_clipped['rivdph']):.3f}"
    )

if not per_delta_results:
    raise RuntimeError("No river reaches processed for any basin in 4_small_deltas")

rivers_clipped_all = gpd.GeoDataFrame(
    pd.concat(per_delta_results, ignore_index=True),
    geometry="geometry",
    crs=delta_domains.crs,
)

summary = (
    rivers_clipped_all.groupby("BasinID2")
    .agg(n_reaches=("reach_id", "count"), mean_rivdph=("rivdph", "mean"), max_rivdph=("rivdph", "max"))
    .sort_values("n_reaches", ascending=False)
)
print("\nFinished per-delta processing for 4_small_deltas")
print(summary)

#%%


#%%

# export the dataset as one file
rivers_clipped_all.to_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\global_SWORD_with_dph_wdth_all_deltas.gpkg", driver="GPKG")



#%%
# 8. VISUALIZE RIVER DEPTHS FOR ALL DELTAS (SEPARATELY)
import matplotlib.pyplot as plt

for basin_id, rivers_delta in rivers_clipped_all.groupby("BasinID2"):
    domain_delta = delta_domains[delta_domains["BasinID2"] == basin_id]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # PLOT 1: Map view - DARK BLUE = DEEPER
    rivers_delta.plot(
        ax=ax1,
        column="rivdph",
        cmap="Blues",
        linewidth=3,
        alpha=0.8,
        legend=True,
        legend_kwds={"label": "Depth (m)", "shrink": 0.8},
    )
    if not domain_delta.empty:
        domain_delta.boundary.plot(ax=ax1, color="black", linewidth=2)
    ax1.set_title(f"Delta {basin_id} River Depths (m)")
    ax1.axis("equal")

    # PLOT 2: Width vs Depth scatter
    scatter = ax2.scatter(
        rivers_delta["rivwth"],
        rivers_delta["rivdph"],
        c=rivers_delta["final_Q2"],
        cmap="viridis",
        s=30,
        alpha=0.7,
    )
    ax2.set_xlabel("Width (m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title(f"Delta {basin_id} Width-Depth Relationship\n(Color = Discharge)")
    plt.colorbar(scatter, ax=ax2, label="Q (m³/s)")

    plt.tight_layout()
    plt.show()

    print(f"\nDelta {basin_id}")
    print(f"Depth range: {rivers_delta['rivdph'].min():.1f} - {rivers_delta['rivdph'].max():.1f} m")
    print(
        f"Widest river: {rivers_delta['rivwth'].max():.0f} m → depth "
        f"{rivers_delta.loc[rivers_delta['rivwth'].idxmax(), 'rivdph']:.1f} m"
    )
    print(
        f"Deepest river: {rivers_delta['rivdph'].max():.1f} m → width "
        f"{rivers_delta.loc[rivers_delta['rivdph'].idxmax(), 'rivwth']:.0f} m"
    )


# %%
import geopandas as gpd
import pandas as pd

# 1. Load the files
sword_rivers = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers.gpkg")
rivers_clipped = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\4_small_rivers_with_dph.gpkg")

def check_dataset(gdf, name):
    print(f"--- Analysis for: {name} ---")
    print(f"Rows/Cols: {gdf.shape}")
    print(f"CRS: {gdf.crs}")
    print(f"Geometry Type: {gdf.geom_type.unique()}")
    print(f"Has NaNs in Geometry? {gdf.geometry.isna().any()}")
    print(f"Is Valid Geometry? {gdf.is_valid.all()}")
    print(f"Bounds: {gdf.total_bounds}") # [minx, miny, maxx, maxy]
    print("\n")

check_dataset(sword_rivers, "Global SWORD")
check_dataset(rivers_clipped, "Clipped Deltas")
# %%

import geopandas as gpd

# 2. Match the Projection (Crucial)
# Move rivers_clipped from Meters back to Degrees to match the Global file
rivers_clipped = rivers_clipped.to_crs(sword_rivers.crs)

# 3. Match Geometry Type (Convert LineString to MultiLineString)
# Some models strictly look for MultiLineString objects
from shapely.geometry import MultiLineString
rivers_clipped['geometry'] = rivers_clipped['geometry'].apply(lambda x: MultiLineString([x]) if x.geom_type == 'LineString' else x)

# 4. Clean up the Schema
# If the model script fails because of "too many columns" or "missing columns",
# we can force rivers_clipped to have the exact same columns as sword_rivers
# plus your new variables (rivdph, etc.)
original_columns = list(sword_rivers.columns)
new_variables = ['rivdph', 'final_Q2'] # add any others you need to keep

# Keep only columns that exist in both + your new data
cols_to_keep = [c for c in rivers_clipped.columns if c in original_columns or c in new_variables]
rivers_clipped = rivers_clipped[cols_to_keep]

# 5. Export
rivers_clipped.to_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\rivers_clipped_formatted.gpkg", driver='GPKG')

print("Formatting complete. The file now matches the CRS and geometry type of the global dataset.")


# %%

