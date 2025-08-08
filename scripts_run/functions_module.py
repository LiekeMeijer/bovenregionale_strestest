import geopandas as gpd
from shapely.geometry import LineString

def load_region_shapefile(region_shapefile, region_name):
    region_gdf = gpd.read_file(region_shapefile)
    filtered_region = region_gdf[region_gdf['name'] == region_name]
    return region_gdf, filtered_region

def load_networks(HWN_network_path, NWB_network_path):
    HWN_network_gdf = gpd.read_file(HWN_network_path)
    NWB_network_gdf = gpd.read_file(NWB_network_path)

    if HWN_network_gdf['NWBWVKID'].dtype != NWB_network_gdf['WVK_ID'].dtype:
        HWN_network_gdf['NWBWVKID'] = HWN_network_gdf['NWBWVKID'].astype(str)
        NWB_network_gdf['WVK_ID'] = NWB_network_gdf['WVK_ID'].astype(str)

    # Add suffixes to all columns except the merge key
    hwn_renamed = HWN_network_gdf.rename(columns={'NWBWVKID': 'WVK_ID'})
    hwn_renamed = hwn_renamed.add_suffix('_HWN')
    hwn_renamed = hwn_renamed.rename(columns={'WVK_ID_HWN': 'WVK_ID'})

    nwb_suffixed = NWB_network_gdf.add_suffix('_NWB')
    nwb_suffixed = nwb_suffixed.rename(columns={'WVK_ID_NWB': 'WVK_ID'})
    
    print("HWN network columns:")
    print(HWN_network_gdf.columns)
    print("NWB network columns:")
    print(NWB_network_gdf.columns)
    print("Joined network columns:")
    
    # Merge the data - this returns a regular DataFrame
    merged_data = nwb_suffixed.merge(
        hwn_renamed,
        on='WVK_ID',
        how='left'
    )
    
    # Convert back to GeoDataFrame with proper geometry column
    Joined_Network = gpd.GeoDataFrame(
        merged_data, 
        geometry=merged_data['geometry_NWB'],
        crs=NWB_network_gdf.crs
    )
    
    # Set the NWB geometry as the primary geometry column
    Joined_Network['geometry'] = Joined_Network['geometry_NWB']
    Joined_Network['match'] = Joined_Network['geometry_HWN'].notna().map({True: 'match', False: 'no_match'})
    
    print(Joined_Network.columns)
    
    return Joined_Network

def filter_network_by_region(joined_network_gdf, filtered_region, code):
    grouped_networks = joined_network_gdf.groupby(code)
    intersecting_nwscodes = []

    for nwscode, group in grouped_networks:
        # Convert group back to GeoDataFrame to access spatial methods
        group_gdf = gpd.GeoDataFrame(group, geometry='geometry')
        if group_gdf.intersects(filtered_region.geometry.iloc[0]).any():
            intersecting_nwscodes.append(nwscode)

    filtered_network_gdf = joined_network_gdf[joined_network_gdf[code].isin(intersecting_nwscodes)]
    return filtered_network_gdf

def create_segments_simple(geometry, segment_length=100):
    if geometry.geom_type != 'LineString':
        gdf = gpd.GeoDataFrame({
            'REF_ID': [1],
            'highway': ['motorway'],
            'geometry': [geometry]
        })
        return gdf

    total_length = geometry.length
    if total_length <= segment_length:
        gdf = gpd.GeoDataFrame({
            'REF_ID': [1],
            'highway': ['motorway'],
            'geometry': [geometry]
        })
        return gdf

    segments = []
    num_segments = int(total_length / segment_length)

    for i in range(num_segments):
        start_dist = i * segment_length
        end_dist = (i + 1) * segment_length
        start_point = geometry.interpolate(start_dist)
        end_point = geometry.interpolate(end_dist)

        try:
            distances = []
            points = []
            step_size = min(10, segment_length / 10)
            current_dist = start_dist

            while current_dist <= end_dist:
                point = geometry.interpolate(current_dist)
                points.append((point.x, point.y))
                current_dist += step_size

            if current_dist - step_size < end_dist:
                end_point = geometry.interpolate(end_dist)
                points.append((end_point.x, end_point.y))

            unique_points = []
            for point in points:
                if not unique_points or point != unique_points[-1]:
                    unique_points.append(point)

            if len(unique_points) >= 2:
                segment = LineString(unique_points)
                segments.append(segment)

        except Exception:
            try:
                segment = LineString([start_point.coords[0], end_point.coords[0]])
                if segment.length > 0:
                    segments.append(segment)
            except Exception:
                continue

    remainder_start = num_segments * segment_length
    if remainder_start < total_length:
        try:
            points = []
            step_size = min(10, (total_length - remainder_start) / 5)
            current_dist = remainder_start

            while current_dist < total_length:
                point = geometry.interpolate(current_dist)
                points.append((point.x, point.y))
                current_dist += step_size

            final_point = geometry.interpolate(total_length)
            points.append((final_point.x, final_point.y))

            unique_points = []
            for point in points:
                if not unique_points or point != unique_points[-1]:
                    unique_points.append(point)

            if len(unique_points) >= 2:
                remainder_segment = LineString(unique_points)
                segments.append(remainder_segment)

        except Exception:
            pass

    # Create GeoDataFrame with ID and highway columns
    if segments:
        gdf = gpd.GeoDataFrame({
            'REF_ID': range(1, len(segments) + 1),
            'highway': ['motorway'] * len(segments),
            'geometry': segments
        })
        return gdf
    else:
        gdf = gpd.GeoDataFrame({
            'REF_ID': [1],
            'highway': ['motorway'],
            'geometry': [geometry]
        })
        return gdf
    

def move_overlay_file(overlay_path, output_path, filename="graph_hazard_overlay.gpkg"):
    """
    Move the graph_hazard_overlay.gpkg file from overlay_path to output_path.
    
    Parameters:
    overlay_path (Path): Source directory path
    output_path (Path): Destination directory path  
    filename (str): Name of the file to move (default: "graph_hazard_overlay.gpkg")
    
    Returns:
    bool: True if file was moved successfully, False otherwise
    """
    import shutil
    from pathlib import Path
    
    source_file = overlay_path / filename
    destination_file = output_path / filename
    
    try:
        if source_file.exists():
            # Ensure destination directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source_file), str(destination_file))
            print(f"Successfully moved {filename} from {overlay_path} to {output_path}")
            return True
        else:
            print(f"File {filename} not found in {overlay_path}")
            return False
            
    except Exception as e:
        print(f"Error moving file: {e}")
        return False