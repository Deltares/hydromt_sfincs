#%%
# Processing rivers 
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

# 1. Load data
delta_domain = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\4_delta_polygons.geojson")
delta_domain = delta_domain[delta_domain['BasinID2'] == 620947].to_crs(epsg=3857)

rivers_sword = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\SWORD_global_unpublished.gpkg", mask=delta_domain).to_crs(epsg=3857)
rivers_lin = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\rivers_lin.gpkg", mask=delta_domain).to_crs(epsg=3857)

# 2. Map boundary inflows
boundary_line = delta_domain.geometry.boundary.iloc[0]
inflow_reaches = rivers_sword[rivers_sword.intersects(boundary_line)].copy()
joined_inflows = gpd.sjoin_nearest(inflow_reaches, rivers_lin, max_distance=100, how='inner')
rivers_sword['inflow_Q2'] = rivers_sword['reach_id'].map(dict(zip(joined_inflows['reach_id'], joined_inflows['Q2']))).fillna(0)

# 3. Create Connection Map
tail_circles = rivers_sword.copy()
tail_circles['geometry'] = tail_circles.geometry.apply(lambda x: Point(x.coords[-1])).buffer(150)
head_points = rivers_sword.copy()
head_points['geometry'] = head_points.geometry.apply(lambda x: Point(x.coords[0]))

# Spatial join to find connections
connections = gpd.sjoin(head_points[['reach_id', 'geometry']], tail_circles[['reach_id', 'geometry']], how='inner', predicate='within')
# Drop geometry and convert to simple DataFrame for the connection map
conn_map = pd.DataFrame(connections[['reach_id_left', 'reach_id_right']])
conn_map = conn_map[conn_map['reach_id_left'] != conn_map['reach_id_right']].astype(int)
conn_map = conn_map.rename(columns={'reach_id_left': 'down_id', 'reach_id_right': 'up_id'})

# 4. Flow Accumulation (Waterfall)
q_dict = rivers_sword.set_index('reach_id')['inflow_Q2'].to_dict()
is_boundary = rivers_sword.set_index('reach_id')['inflow_Q2'] > 0
widths = rivers_sword.set_index('reach_id')['width'].to_dict()

for _ in range(100):
    next_q = {rid: val for rid, val in q_dict.items() if is_boundary.get(rid, False)}
    for up_id, group in conn_map.groupby('up_id'):
        parent_flow = q_dict.get(up_id, 0)
        if parent_flow <= 0: continue
        
        down_ids = group['down_id'].unique()
        total_w = sum(widths.get(d_id, 0) for d_id in down_ids)
        for d_id in down_ids:
            if not is_boundary.get(d_id, False):
                share = widths.get(d_id, 0) / total_w if total_w > 0 else (1.0 / len(down_ids))
                next_q[d_id] = next_q.get(d_id, 0) + (parent_flow * share)
    q_dict.update(next_q)

# rename width variable to match sfincs requirements
rivers_sword = rivers_sword.rename(columns={'width': 'rivwth'})

# Calculate depth of river segments 
rivers_sword['final_Q2'] = rivers_sword['reach_id'].map(q_dict).fillna(0)
rivers_clipped = rivers_sword[rivers_sword['final_Q2'] > 1e-4].copy()

a = 0.27
b = 0.30
rivers_clipped['rivdph'] = (a * (rivers_clipped['final_Q2']**b)).clip(lower=0.1)


#%%
print(rivers_clipped[['inflow_Q2', 'reach_id', 'final_Q2', 'rivwth', 'rivdph']])

# %%

rivers_sword_old = gpd.read_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers.gpkg", mask=delta_domain).to_crs(epsg=3857)

# add the riv_dph variable to this SWORD dataset 
rivers_sword_old = gpd.sjoin_nearest(rivers_sword_old, rivers_clipped[['rivdph', 'geometry']], how='left', max_distance=100)
# Drop duplicates if multiple segments were found within distance and remove join index
rivers_sword_old = rivers_sword_old.loc[~rivers_sword_old.index.duplicated(keep='first')]
if 'index_right' in rivers_sword_old.columns:
    rivers_sword_old = rivers_sword_old.drop(columns=['index_right'])

print(rivers_sword_old.columns)

# export as gpkg
rivers_sword_old.to_file(r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers_with_dph.gpkg", driver='GPKG')







# %%
