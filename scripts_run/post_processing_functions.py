import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from typing import List, Dict, Tuple, Optional

def load_traffic_centers(traffic_centers_file: str, buffer_distance: int = 1000) -> gpd.GeoDataFrame:
    """
    Load traffic centers and create buffered geometries
    """
    print("Loading traffic centers...")
    
    df = pd.read_excel(Path(traffic_centers_file))
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.crs = "EPSG:4326"
    
    # Project to RD New (Netherlands coordinate system)
    gdf_projected = gdf.to_crs("EPSG:28992")
    
    # Create buffered version
    gdf_buffered = gdf_projected.copy()
    gdf_buffered['geometry'] = gdf_projected.geometry.buffer(buffer_distance)
    
    print(f"Loaded {len(gdf_buffered)} traffic centers with {buffer_distance}m buffers")
    return gdf_buffered

def get_flood_map_files(region: str, base_path: str) -> List[Path]:
    """
    Get all flood map files for a specific region
    
    Args:
      
        
    Returns:
        List of flood map file paths
    """
    flood_map_folder = Path(base_path) / region / "Basisscenario" / "waterdiepte"
    
    if not flood_map_folder.exists():
        print(f"Warning: Folder {flood_map_folder} does not exist")
        return []
    
    flood_map_files = (
        list(flood_map_folder.glob("*.tif")) + 
        list(flood_map_folder.glob("*.tiff"))
    )
    
    print(f"Found {len(flood_map_files)} flood map files for region {region}")
    return flood_map_files


def extract_flood_depths_from_raster(flood_map_path: Path, gdf_buffered: gpd.GeoDataFrame) -> Dict:
    """
    Extract flood depths from a single flood map for all traffic centers
    
    Args:
        flood_map_path: Path to the flood map raster file
        gdf_buffered: GeoDataFrame with buffered traffic centers
        
    Returns:
        Dictionary with flood depth statistics
    """
    try:
        with rasterio.open(flood_map_path) as flood_raster:
            print(f"Processing {flood_map_path.name}...")
            
            # Transform geometries to match raster CRS
            gdf_raster_crs = gdf_buffered.to_crs(flood_raster.crs)
            
            flood_depths = []
            
            for idx, row in gdf_raster_crs.iterrows():
                geom = row.geometry
                
                try:
                    # Sample raster within the buffered area
                    out_image, out_transform = mask(
                        flood_raster, 
                        [geom], 
                        crop=True, 
                        nodata=flood_raster.nodata
                    )
                    
                    # Get valid pixels (excluding nodata)
                    valid_pixels = out_image[out_image != flood_raster.nodata]
                    
                    if len(valid_pixels) > 0:
                        # Calculate statistics
                        max_depth = float(np.max(valid_pixels))
                        mean_depth = float(np.mean(valid_pixels))
                        min_depth = float(np.min(valid_pixels))
                        std_depth = float(np.std(valid_pixels))
                        pixel_count = len(valid_pixels)
                        
                        # Additional statistics
                        flooded_pixels = (valid_pixels > 0).sum()
                        flood_percentage = (flooded_pixels / pixel_count * 100) if pixel_count > 0 else 0
                    else:
                        max_depth = mean_depth = min_depth = std_depth = 0.0
                        pixel_count = flooded_pixels = flood_percentage = 0
                    
                    flood_depths.append({
                        'max_flood_depth': max_depth,
                        'mean_flood_depth': mean_depth,
                        'min_flood_depth': min_depth,
                        'std_flood_depth': std_depth,
                        'pixel_count': pixel_count,
                        'flooded_pixels': flooded_pixels,
                        'flood_percentage': flood_percentage
                    })
                    
                except Exception as e:
                    print(f"Error processing geometry {idx}: {e}")
                    flood_depths.append({
                        'max_flood_depth': -9999,
                        'mean_flood_depth': -9999,
                        'min_flood_depth': -9999,
                        'std_flood_depth': -9999,
                        'pixel_count': 0,
                        'flooded_pixels': 0,
                        'flood_percentage': 0
                    })
            
            return {
                'file_name': flood_map_path.stem,
                'file_path': str(flood_map_path),
                'region': None,  # Will be set by calling function
                'depths': flood_depths
            }
            
    except Exception as e:
        print(f"Error processing flood map {flood_map_path}: {e}")
        return None

def add_flood_columns_to_gdf(gdf: gpd.GeoDataFrame, flood_result: Dict, region: str) -> gpd.GeoDataFrame:
    """
    Add flood depth columns to GeoDataFrame
    
    Args:
        gdf: GeoDataFrame to add columns to
        flood_result: Result from extract_flood_depths_from_raster
        region: Region name
        
    Returns:
        GeoDataFrame with added flood columns
    """
    if flood_result is None:
        return gdf
    
    file_suffix = f"{region}_{flood_result['file_name']}"
    
    # Add all flood depth statistics as columns
    gdf[f'max_flood_depth_{file_suffix}'] = [d['max_flood_depth'] for d in flood_result['depths']]
    gdf[f'mean_flood_depth_{file_suffix}'] = [d['mean_flood_depth'] for d in flood_result['depths']]
    gdf[f'min_flood_depth_{file_suffix}'] = [d['min_flood_depth'] for d in flood_result['depths']]
    gdf[f'std_flood_depth_{file_suffix}'] = [d['std_flood_depth'] for d in flood_result['depths']]
    gdf[f'pixel_count_{file_suffix}'] = [d['pixel_count'] for d in flood_result['depths']]
    gdf[f'flooded_pixels_{file_suffix}'] = [d['flooded_pixels'] for d in flood_result['depths']]
    gdf[f'flood_percentage_{file_suffix}'] = [d['flood_percentage'] for d in flood_result['depths']]
    
    # Optionally add a column to track which region/file was used
    gdf['flood_source'] = f"{region}_{flood_result['file_name']}"
    
    return gdf

def process_region_flood_maps(region: str, gdf_buffered: gpd.GeoDataFrame, hazard_maps_base_path: str) -> Tuple[gpd.GeoDataFrame, List[Dict]]:
    """
    Process all flood maps for a single region
    
    Args:
        region: Region name
        gdf_buffered: GeoDataFrame with buffered traffic centers
        hazard_maps_base_path: Base path to hazard maps
        
    Returns:
        Tuple of (updated GeoDataFrame, list of flood results)
    """
    print(f"\n=== Processing region: {region} ===")
    
    gdf_result = gdf_buffered.copy()
    region_results = []
    
    flood_map_files = get_flood_map_files(region, hazard_maps_base_path)
    
    for flood_map_file in flood_map_files:
        flood_result = extract_flood_depths_from_raster(flood_map_file, gdf_buffered)
        
        if flood_result is not None:
            flood_result['region'] = region
            gdf_result = add_flood_columns_to_gdf(gdf_result, flood_result, region)
            region_results.append(flood_result)
    
    return gdf_result, region_results

def process_all_regions(gdf_buffered: gpd.GeoDataFrame, region_list: List[str], hazard_maps_base_path: str) -> Tuple[gpd.GeoDataFrame, List[Dict]]:
    """
    Process flood maps for all regions
    
    Args:
        gdf_buffered: GeoDataFrame with buffered traffic centers
        region_list: List of region names
        hazard_maps_base_path: Base path to hazard maps
        
    Returns:
        Tuple of (GeoDataFrame with all flood data, list of all results)
    """
    result_gdf = gdf_buffered.copy()
    all_results = []
    
    for region in region_list:
        # Get the current state of the GeoDataFrame
        result_gdf, region_results = process_region_flood_maps(region, result_gdf, hazard_maps_base_path)
        all_results.extend(region_results)
    
    return result_gdf, all_results

def save_results(gdf_with_floods: gpd.GeoDataFrame, output_directory: str) -> Tuple[Path, Path]:
    """
    Save results to files
    
    Args:
        gdf_with_floods: GeoDataFrame with flood depth results
        output_directory: Directory to save results
        
    Returns:
        Tuple of (gpkg_path, excel_path)
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary columns with highest values from all region-specific columns
    print("Creating summary columns with maximum values across all regions...")
    
    # Find all region-specific flood depth columns
    max_flood_cols = [col for col in gdf_with_floods.columns if col.startswith('max_flood_depth_')]
    mean_flood_cols = [col for col in gdf_with_floods.columns if col.startswith('mean_flood_depth_')]
    min_flood_cols = [col for col in gdf_with_floods.columns if col.startswith('min_flood_depth_')]
    pixel_count_cols = [col for col in gdf_with_floods.columns if col.startswith('pixel_count_')]
    flooded_pixels_cols = [col for col in gdf_with_floods.columns if col.startswith('flooded_pixels_')]
    flood_percentage_cols = [col for col in gdf_with_floods.columns if col.startswith('flood_percentage_')]
    
    # Create summary columns by taking the maximum value across all region columns
    if max_flood_cols:
        # Filter out -9999 values when calculating max
        max_values = gdf_with_floods[max_flood_cols].replace(-9999, np.nan).max(axis=1)
        gdf_with_floods['max_flood_depth'] = max_values.fillna(-9999)
    else:
        gdf_with_floods['max_flood_depth'] = -9999
    
    if mean_flood_cols:
        # For mean flood depth, take the maximum of the means
        mean_values = gdf_with_floods[mean_flood_cols].replace(-9999, np.nan).max(axis=1)
        gdf_with_floods['mean_flood_depth'] = mean_values.fillna(-9999)
    else:
        gdf_with_floods['mean_flood_depth'] = -9999
    
    if min_flood_cols:
        # For min flood depth, take the maximum of the mins (highest minimum depth)
        min_values = gdf_with_floods[min_flood_cols].replace(-9999, np.nan).max(axis=1)
        gdf_with_floods['min_flood_depth'] = min_values.fillna(-9999)
    else:
        gdf_with_floods['min_flood_depth'] = -9999
    
    if pixel_count_cols:
        # Take maximum pixel count
        gdf_with_floods['pixel_count'] = gdf_with_floods[pixel_count_cols].max(axis=1)
    else:
        gdf_with_floods['pixel_count'] = 0
    
    if flooded_pixels_cols:
        # Take maximum flooded pixels
        gdf_with_floods['flooded_pixels'] = gdf_with_floods[flooded_pixels_cols].max(axis=1)
    else:
        gdf_with_floods['flooded_pixels'] = 0
    
    if flood_percentage_cols:
        # Take maximum flood percentage
        gdf_with_floods['flood_percentage'] = gdf_with_floods[flood_percentage_cols].max(axis=1)
    else:
        gdf_with_floods['flood_percentage'] = 0
    
    # Add a column to indicate which region/file had the maximum flood depth
    if max_flood_cols:
        max_source_cols = []
        for idx, row in gdf_with_floods.iterrows():
            max_depth = row['max_flood_depth']
            if max_depth != -9999:
                # Find which column had this maximum value
                for col in max_flood_cols:
                    if row[col] == max_depth:
                        # Extract region and file info from column name
                        source_info = col.replace('max_flood_depth_', '')
                        max_source_cols.append(source_info)
                        break
                else:
                    max_source_cols.append('Unknown')
            else:
                max_source_cols.append('No_flood_data')
        
        gdf_with_floods['max_flood_source'] = max_source_cols
    
    print(f"Summary statistics:")
    print(f"  - Traffic centers with flood data: {(gdf_with_floods['max_flood_depth'] != -9999).sum()}")
    print(f"  - Traffic centers with flooding (>0m): {(gdf_with_floods['max_flood_depth'] > 0).sum()}")
    print(f"  - Maximum flood depth found: {gdf_with_floods['max_flood_depth'].max():.2f}m")
    
    # Save as GeoPackage
    gpkg_path = output_dir / "traffic_centers_with_flood_depths.gpkg"
    gdf_with_floods.to_file(gpkg_path, driver="GPKG")
    
    # Save as Excel (without geometry)
    excel_df = gdf_with_floods.drop(columns=['geometry'])
    excel_path = output_dir / "traffic_centers_with_flood_depths.xlsx"
    excel_df.to_excel(excel_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"  - GeoPackage: {gpkg_path}")
    print(f"  - Excel: {excel_path}")
    
    return gpkg_path, excel_path

import networkx as nx
def cluster_connected(gdf):
    """
    Spatially cluster AFR ramps that are connected (touching).
    Returns a copy of gdf with a 'cluster_id' column.
    """
    import networkx as nx
    G = nx.Graph()
    for idx, geom in gdf.geometry.items():
        G.add_node(idx)
    for idx1, geom1 in gdf.geometry.items():
        for idx2, geom2 in gdf.geometry.items():
            if idx1 < idx2 and geom1.touches(geom2):
                G.add_edge(idx1, idx2)
    clusters = list(nx.connected_components(G))
    gdf = gdf.copy()
    gdf["cluster_id"] = -1
    for cluster_idx, cluster_nodes in enumerate(clusters):
        gdf.loc[gdf.index.isin(cluster_nodes), "cluster_id"] = cluster_idx
    return gdf


from shapely.ops import nearest_points
def aggregate_clusters_to_points(gdf, aggregation_column, method="mean"):
    """
    For each cluster (from 'cluster_id'), create a point (centroid snapped to nearest line in cluster)
    and aggregate the specified column using the given method ('mean', 'median', 'max').
    
    Args:
        gdf: GeoDataFrame with 'cluster_id' column
        aggregation_column: Name of the column to aggregate
        method: Aggregation method ('mean', 'median', 'max')
        
    Returns:
        GeoDataFrame with one point per cluster and aggregated column
    """
    clusters = []
    for cluster_id, group in gdf.groupby("cluster_id"):
        # Aggregate geometry: union of all geometries in cluster
        union_geom = group.geometry.unary_union
        # Centroid of union
        centroid = union_geom.centroid
        # Snap centroid to nearest point on the unioned geometry (line)
        snapped_point, _ = nearest_points(union_geom, centroid)
        # Aggregate values
        values = group[aggregation_column].dropna()
        if len(values) == 0:
            agg_value = np.nan
        elif method == "mean":
            agg_value = values.mean()
        elif method == "median":
            agg_value = values.median()
        elif method == "max":
            agg_value = values.max()
        else:
            raise ValueError("Unknown aggregation method: {}".format(method))
        clusters.append({
            "cluster_id": cluster_id,
            "geometry": snapped_point,
            aggregation_column: agg_value
        })
    return gpd.GeoDataFrame(clusters, geometry="geometry", crs=gdf.crs)


def aggregate_line_sections(network_gdf, column, agg_method='mean'):
    """
    Dissolves segments based on the specified column and aggregates EV columns using the specified method.

    Parameters:
        network_gdf (GeoDataFrame): Input GeoDataFrame with 'id_NWB' and EV columns.
        column (str): Column name to dissolve by.
        agg_method (str): Aggregation method: 'mean', 'median', 'min', or 'max'.

    Returns:
        GeoDataFrame: Aggregated GeoDataFrame.
    """
    

    # Select EV columns automatically
    ev_columns = [col for col in network_gdf.columns if col.startswith('EV')]

    agg_funcs = {
        'mean': np.mean,
        'median': np.median,
        'min': np.min,
        'max': np.max
    }
    if agg_method not in agg_funcs:
        raise ValueError("agg_method must be one of: 'mean', 'median', 'min', 'max'")

    agg_dict = {col: agg_funcs[agg_method] for col in ev_columns}

    # Dissolve by 'id_NWB' and aggregate EV columns
    aggregated_gdf = network_gdf.dissolve(by=column, aggfunc=agg_dict)
    aggregated_gdf = aggregated_gdf.reset_index()  # Restore id_NWB as a column

    return aggregated_gdf

def get_z_height_optimized(lines_gdf, points_gdf, threshold=50.0):
    # Create spatial index for points
    points_sindex = points_gdf.sindex
    
    # Initialize result array
    z_heights = np.full(len(lines_gdf), np.nan)
    
    for idx, line_geom in enumerate(lines_gdf.geometry):
        # Use spatial index to get potential matches
        bounds = line_geom.bounds
        # Expand bounds by threshold
        expanded_bounds = (
            bounds[0] - threshold, bounds[1] - threshold,
            bounds[2] + threshold, bounds[3] + threshold
        )
        
        # Get potential point indices using spatial index
        possible_matches_index = list(points_sindex.intersection(expanded_bounds))
        
        if possible_matches_index:
            # Get the actual geometries
            possible_matches = points_gdf.iloc[possible_matches_index]
            
            # Calculate distances (vectorized)
            distances = possible_matches.geometry.distance(line_geom)
            
            # Filter points within threshold
            within_threshold = distances <= threshold
            
            if within_threshold.any():
                # Get Z values of points within threshold and calculate mean
                z_values = possible_matches.loc[within_threshold, 'Z']
                z_heights[idx] = z_values.mean()  # Use mean instead of nearest
    
    return z_heights


def calculate_overlay_percentages(lines_gdf, bridges_gdf, tunnels_gdf):
    """
    Calculates the percentage of each line segment overlapped by tunnels and bridges.
    Updates lines_gdf in-place with 'tunnel_percentage' and 'bridge_percentage' columns.
    """
    # Ensure all use the same CRS
    bridges_gdf = bridges_gdf.to_crs(lines_gdf.crs)
    tunnels_gdf = tunnels_gdf.to_crs(lines_gdf.crs)

    # Initialize percentage columns
    lines_gdf['tunnel_percentage'] = 0.0
    lines_gdf['bridge_percentage'] = 0.0

    # Calculate tunnel overlay percentages (avoiding double counting)
    for idx, line_row in lines_gdf.iterrows():
        line_geom = line_row.geometry
        intersecting_tunnels = tunnels_gdf[tunnels_gdf.geometry.intersects(line_geom)]
        if not intersecting_tunnels.empty:
            tunnel_union = intersecting_tunnels.geometry.unary_union
            intersection = line_geom.intersection(tunnel_union)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    overlap_length = intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    overlap_length = sum(geom.length for geom in intersection.geoms)
                else:
                    overlap_length = 0
                percentage = (overlap_length / line_geom.length) * 100
                lines_gdf.loc[idx, 'tunnel_percentage'] = percentage

    # Calculate bridge overlay percentages (avoiding double counting)
    for idx, line_row in lines_gdf.iterrows():
        line_geom = line_row.geometry
        intersecting_bridges = bridges_gdf[bridges_gdf.geometry.intersects(line_geom)]
        if not intersecting_bridges.empty:
            bridge_union = intersecting_bridges.geometry.unary_union
            intersection = line_geom.intersection(bridge_union)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    overlap_length = intersection.length
                elif intersection.geom_type == 'MultiLineString':
                    overlap_length = sum(geom.length for geom in intersection.geoms)
                else:
                    overlap_length = 0
                percentage = (overlap_length / line_geom.length) * 100
                lines_gdf.loc[idx, 'bridge_percentage'] = percentage

    return lines_gdf

def Filter_and_aggregate_flooded_segments_damage(gdf, output_dir,ex, dissolve_col='NETWERKSCH'):
    """
    Aggregates flooded segments and saves results to file.

    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame with flood columns.
        output_dir (Path): Output directory for saving files.
        dissolve_col (str): Column to dissolve by (default: 'NETWERKSCH').
    """
    # Step 1: Initialize the score column
    gdf['Flood_uncertainty'] = 0

    # Step 2: Add 1 if EV1_fr < 0.2
    gdf.loc[gdf['F_EV1_fr'] < 0.2, 'Flood_uncertainty'] += 1

    # Step 3: Add 1 if both EV1_me and EV1_ma <= 0.3
    gdf.loc[(gdf['F_EV1_me'] <= 0.3) & (gdf['F_EV1_ma'] <= 0.3), 'Flood_uncertainty'] += 1

    # Step 2b: Add 1 if EV1_fr is between 0.2 and 0.3 and EV1_ma < 0.5
    gdf.loc[(gdf['F_EV1_fr'] >= 0.2) & (gdf['F_EV1_fr'] < 0.3) & (gdf['F_EV1_ma'] < 0.5), 'Flood_uncertainty'] += 1

    # Step 4: Define the isolation check
    def is_isolated(index, flooded_series):
        if not flooded_series.iloc[index]:
            return False
        neighbors = []
        if index > 0:
            neighbors.append(flooded_series.iloc[index - 1])
        if index < len(flooded_series) - 1:
            neighbors.append(flooded_series.iloc[index + 1])
        return not any(neighbors)

    # Step 5: Apply isolation logic and update score
    gdf['is_flooded'] = gdf['Flood_uncertainty'] > 0
    gdf['is_isolated'] = [is_isolated(i, gdf['is_flooded']) for i in range(len(gdf))]
    gdf.loc[gdf['is_isolated'], 'Flood_uncertainty'] += 1

    gdf = gdf.set_crs("EPSG:28992", allow_override=True)
    gdf.to_file(output_dir.joinpath(f"{ex}flooded_segments_case_based.gpkg"), driver='GPKG')

    # Identify dam_EV columns
    ev_cols = [col for col in gdf.columns if col.startswith('EV')]
    dam_ev_cols = [col for col in gdf.columns if col.startswith('dam_EV')]

    # Create new columns
    for col in dam_ev_cols:
        gdf[f'lower_{col}'] = gdf[col]
        gdf[f'upper_{col}'] = gdf[col]

    # Apply condition to zero out lower_dam_EV
    condition = (
        (gdf['Flood_uncertainty'] > 0) |
        (gdf['bridge_percentage'] > 60) |
        (gdf['tunnel_percentage'] > 60)
    )

    condition2 = (
        (gdf['Flood_uncertainty'] > 0) 
    )

    for col in dam_ev_cols:
        gdf.loc[condition, f'lower_{col}'] = 0

    for col in dam_ev_cols:
        gdf.loc[condition2, f'upper_{col}'] = 0

    # Calculate flooded length per segment
    #gdf['flooded_length'] = gdf['F_EV1_fr'] * gdf['length']

    # Define aggregation dictionary
    agg_dict = {
        'length': 'sum',
    **{f'lower_{col}': 'sum' for col in dam_ev_cols},
    **{f'upper_{col}': 'sum' for col in dam_ev_cols},
    **{col: ['mean', 'max', 'median'] for col in ev_cols}  # if you still want to include EV columns
    }

    # Dissolve geometries and aggregate
    gdf_dissolved = gdf.dissolve(by=dissolve_col, aggfunc=agg_dict, as_index=False)

    # Flatten multi-level columns
    gdf_dissolved.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in gdf_dissolved.columns
    ]

    # Rename for clarity
    gdf_dissolved = gdf_dissolved.rename(columns={
        'length_sum': 'total_length',
        'flooded_length_sum': 'flooded_length'
    })

    # Calculate fraction flooded
    #gdf_dissolved['fraction_flooded'] = gdf_dissolved['flooded_length'] / gdf_dissolved['total_length']

    # Convert to GeoDataFrame and set CRS
    gdf_agg = gpd.GeoDataFrame(gdf_dissolved, geometry='geometry', crs=gdf.crs)
    gdf_agg = gdf_agg.set_crs("EPSG:28992", allow_override=True)

    # Save to file
    gdf_agg = gdf_agg[gdf_agg.is_valid]

    gdf_agg.to_file(output_dir.joinpath(f"{ex}Aggregated_flooded_segments.gpkg"), driver='GPKG')

    return gdf_agg

def Filter_and_aggregate_flooded_segments_exposure(gdf, output_dir,ex, dissolve_col='NETWERKSCH'):
    """
    Aggregates flooded segments and saves results to file.

    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame with flood columns.
        output_dir (Path): Output directory for saving files.
        dissolve_col (str): Column to dissolve by (default: 'NETWERKSCH').
    """
    # Step 1: Initialize the score column
    gdf['Flood_uncertainty'] = 0

    # Step 2: Add 1 if EV1_fr < 0.2
    gdf.loc[gdf['EV1_fr'] < 0.2, 'Flood_uncertainty'] += 1

    # Step 3: Add 1 if both EV1_me and EV1_ma <= 0.3
    gdf.loc[(gdf['EV1_me'] <= 0.3) & (gdf['EV1_ma'] <= 0.3), 'Flood_uncertainty'] += 1

    # Step 2b: Add 1 if EV1_fr is between 0.2 and 0.3 and EV1_ma < 0.5
    gdf.loc[(gdf['EV1_fr'] >= 0.2) & (gdf['EV1_fr'] < 0.3) & (gdf['EV1_ma'] < 0.5), 'Flood_uncertainty'] += 1

    # Step 4: Define the isolation check
    def is_isolated(index, flooded_series):
        if not flooded_series.iloc[index]:
            return False
        neighbors = []
        if index > 0:
            neighbors.append(flooded_series.iloc[index - 1])
        if index < len(flooded_series) - 1:
            neighbors.append(flooded_series.iloc[index + 1])
        return not any(neighbors)

    # Step 5: Apply isolation logic and update score
    gdf['is_flooded'] = gdf['Flood_uncertainty'] > 0
    gdf['is_isolated'] = [is_isolated(i, gdf['is_flooded']) for i in range(len(gdf))]
    gdf.loc[gdf['is_isolated'], 'Flood_uncertainty'] += 1

    gdf = gdf.set_crs("EPSG:28992", allow_override=True)
    gdf.to_file(output_dir.joinpath(f"{ex}flooded_segments_case_based.gpkg"), driver='GPKG')

    ev_cols = [col for col in gdf.columns if col.startswith('EV')]
    gdf.loc[
        (gdf['Flood_uncertainty'] > 0) | 
        (gdf['bridge_percentage'] > 60) | 
        (gdf['tunnel_percentage'] > 60), 
        ev_cols
    ] = 0

    # Calculate flooded length per segment
    gdf['flooded_length'] = gdf['EV1_fr'] * gdf['length']
    gdf['bridge_length'] = (gdf['bridge_percentage'] / 100) * gdf['length']
    gdf['tunnel_length'] = (gdf['tunnel_percentage'] / 100) * gdf['length']

    # Define aggregation dictionary
    agg_dict = {
        'length': 'sum',
        'flooded_length': 'sum',
        'bridge_length': 'sum',
        'tunnel_length': 'sum',
        **{col: ['mean', 'max', 'median'] for col in ev_cols}
    }

    # Dissolve geometries and aggregate
    gdf_dissolved = gdf.dissolve(by=dissolve_col, aggfunc=agg_dict, as_index=False)

    # Flatten multi-level columns
    gdf_dissolved.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in gdf_dissolved.columns
    ]

    # Rename for clarity
    gdf_dissolved = gdf_dissolved.rename(columns={
        'length_sum': 'total_length',
        'flooded_length_sum': 'flooded_length'
    })

    # Calculate fraction flooded
    gdf_dissolved['fraction_flooded'] = gdf_dissolved['flooded_length'] / gdf_dissolved['total_length']

    # Convert to GeoDataFrame and set CRS
    gdf_agg = gpd.GeoDataFrame(gdf_dissolved, geometry='geometry', crs=gdf.crs)
    gdf_agg = gdf_agg.set_crs("EPSG:28992", allow_override=True)

    # Save to file
    gdf_agg.to_file(output_dir.joinpath(f"{ex}Aggregated_flooded_segments.gpkg"), driver='GPKG')

    return gdf_agg

