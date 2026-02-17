#%%
# Processing rivers 
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

# 1. Load data
def river_depth_estimation(delta_domain, rivers_sword_network, rivers_lin, rivers_sword_old):
    """
    Estimate river depth for river segments in a delta domain using hydrological relationships.
    This function processes river network data to calculate river depths based on discharge (Q2) values.
    It maps boundary inflows from a linear river dataset to the SWORD dataset, traces flow accumulation
    through the river network, and estimates depth using a power-law relationship.
    Parameters
    ----------
    delta_domain : GeoDataFrame
        A GeoDataFrame representing the delta domain boundary. Used to identify inflow reaches
        that intersect with the domain boundary.
    rivers_sword_network : GeoDataFrame
        A GeoDataFrame containing SWORD river network data with columns 'reach_id' and 'width'.
        The geometry column holds LineString objects representing river centerlines/segments.
    rivers_lin : GeoDataFrame
        A GeoDataFrame containing linear river data with column 'Q2' (discharge values).
        Used to assign flow values to boundary inflows.
    rivers_sword_old : GeoDataFrame
        The original SWORD dataset to which depth and discharge values will be added.
        The geometry column holds LineString objects of original river segments.

    Returns
    -------
    GeoDataFrame
        Modified rivers_sword_old with added columns 'rivdph' (river depth in meters) and
        'final_Q2' (accumulated discharge). Rows with final_Q2 <= 1e-4 are excluded.
        The geometry column holds LineString objects of river segments.
    Notes
    -----
    - All inputs are reprojected to EPSG:3857 (Web Mercator) for spatial operations.
    - The geometry column in all inputs/outputs holds LineString geometries representing river centerlines.
    - Flow accumulation is computed iteratively (100 iterations) through the river network
      based on spatial connectivity within 150m buffers.
    - River depth is clipped to a minimum value of 0.1 meters.
    """
    # 1. Reproject all data to a common CRS (EPSG:3857)
    delta_domain = delta_domain.to_crs(epsg=3857)
    rivers_sword_network = rivers_sword_network.to_crs(epsg=3857)
    rivers_lin = rivers_lin.to_crs(epsg=3857)
    rivers_sword_old = rivers_sword_old.to_crs(epsg=3857)

    # 2. Map boundary inflows
    boundary_line = delta_domain.geometry.boundary.iloc[0]
    inflow_reaches = rivers_sword_network[rivers_sword_network.intersects(boundary_line)].copy()
    joined_inflows = gpd.sjoin_nearest(inflow_reaches, rivers_lin, max_distance=100, how='inner')
    rivers_sword_network['inflow_Q2'] = rivers_sword_network['reach_id'].map(dict(zip(joined_inflows['reach_id'], joined_inflows['Q2']))).fillna(0)

    # 3. Create Connection Map
    tail_circles = rivers_sword_network.copy()
    tail_circles['geometry'] = tail_circles.geometry.apply(lambda x: Point(x.coords[-1])).buffer(150)
    head_points = rivers_sword_network.copy()
    head_points['geometry'] = head_points.geometry.apply(lambda x: Point(x.coords[0]))

    # Spatial join to find connections
    connections = gpd.sjoin(head_points[['reach_id', 'geometry']], tail_circles[['reach_id', 'geometry']], how='inner', predicate='within')
    # Drop geometry and convert to simple DataFrame for the connection map
    conn_map = pd.DataFrame(connections[['reach_id_left', 'reach_id_right']])
    conn_map = conn_map[conn_map['reach_id_left'] != conn_map['reach_id_right']].astype(int)
    conn_map = conn_map.rename(columns={'reach_id_left': 'down_id', 'reach_id_right': 'up_id'})

    # 4. Flow Accumulation (Waterfall)
    q_dict = rivers_sword_network.set_index('reach_id')['inflow_Q2'].to_dict()
    is_boundary = rivers_sword_network.set_index('reach_id')['inflow_Q2'] > 0
    widths = rivers_sword_network.set_index('reach_id')['width'].to_dict()

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
    rivers_sword_network = rivers_sword_network.rename(columns={'width': 'rivwth'})

    # Calculate depth of river segments 
    rivers_sword_network['final_Q2'] = rivers_sword_network['reach_id'].map(q_dict).fillna(0)
    rivers_clipped = rivers_sword_network[rivers_sword_network['final_Q2'] > 1e-4].copy()

    rivers_clipped['rivdph'] = (0.27 * (rivers_clipped['final_Q2']**0.30)).clip(lower=0.1)

    # add the riv_dph and final_Q2 variables to this SWORD dataset 
    rivers_clipped = gpd.sjoin_nearest(rivers_sword_old, rivers_clipped[['rivdph', 'final_Q2', 'geometry']], how='left', max_distance=100)
    
    # Drop duplicates if multiple segments were found within distance and remove join index
    rivers_clipped = rivers_clipped.loc[~rivers_clipped.index.duplicated(keep='first')]
    if 'index_right' in rivers_clipped.columns:
        rivers_clipped = rivers_clipped.drop(columns=['index_right'])
    
    return rivers_clipped

# %%
# if __name__ == "__main__":
#     # basin_id = 620947
#     basin_id = 4267691
#     delta_domain_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\4_delta_polygons.geojson"
#     rivers_sword_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\SWORD_global_unpublished.gpkg"
#     rivers_lin_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\rivers_lin.gpkg"
#     rivers_sword_old_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers.gpkg"
#     output_path = r"C:\PhD\SFINCS\SFINCS_cloned\input\Global_rivers_with_dph.gpkg"

#     delta_domain = gpd.read_file(delta_domain_path)
#     delta_domain = delta_domain[delta_domain['BasinID2'] == basin_id]

#     rivers_sword = gpd.read_file(rivers_sword_path, mask=delta_domain)
#     rivers_lin = gpd.read_file(rivers_lin_path, mask=delta_domain)
#     rivers_sword_old = gpd.read_file(rivers_sword_old_path, mask=delta_domain)

#     rivers_with_dph = river_depth_estimation(
#         delta_domain=delta_domain,
#         rivers_sword=rivers_sword,
#         rivers_lin=rivers_lin,
#         rivers_sword_old=rivers_sword_old
#     )

#     print(rivers_with_dph.columns)
#     # export as gpkg
#     rivers_with_dph.to_file(output_path, driver='GPKG')




# %%
