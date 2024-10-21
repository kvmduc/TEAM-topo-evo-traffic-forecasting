import numpy as np
import networkx as nx
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import folium

import io
from PIL import Image


SEED = 6789
graph_dir = '/home/s223719687/python_project/traffic_evolve/data/PeMS3_Expand/graph'



nodes_colors = []
edges_colors = []
pos = None

meta_df = pd.read_csv('/home/s223719687/python_project/traffic_evolve/cl_traffic_coat/sensor3_lat_long.csv', sep='\t')

def extract_pos(df, nCount):
    pos = {}
    for i in range (0,nCount):
        value = df[(df.loc[:,'index']) == i]
        lat = value.Latitude.values[0]
        long = value.Longitude.values[0]
        pos[i] = np.array([long, lat])
    return pos

def scale_comp(g, old_pos):
    new_pos = {}
    for c in nx.connected_components(g):
        component_pos = {}
        for node in c:
            component_pos[node] = old_pos[node]
        
        rescaled_component_pos = nx.spring_layout(g, pos=component_pos, scale=10, iterations=10, seed=SEED)
        for node in rescaled_component_pos.keys():
            new_pos[node] = rescaled_component_pos[node]
    return new_pos        
        

def get_dist(G):
    df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
    for row, data in nx.shortest_path_length(G):
        for col, dist in data.items():
            df.loc[row,col] = dist

    df = df.fillna(df.max().max())
    return df

def calculate_centroid(df):
    """
    Calculate the centroid (geometric center) of a set of points.
    
    Parameters:
    - df: DataFrame containing lat and long columns
    
    Returns:
    - A tuple (centroid_lat, centroid_lon).
    """
    centroid_lat = df['Latitude'].mean()
    centroid_lon = df['Longitude'].mean()
    
    return centroid_lat, centroid_lon

def generate_sensor_map(traffic_sensors_df, nCount, new_node_list, removed_node_list):
    """
    Generate a map showing the traffic sensors and weather sensors.
    
    Parameters:
    - traffic_sensors_df: DataFrame containing traffic sensor locations.
    - new_node_list: List contain new nodes.
    - removed_node_list: List contain removed nodes.
    
    Returns:
    - A folium map object.
    """

    centroid_lat, centroid_lon = calculate_centroid(traffic_sensors_df)
    
    # Create a base map
    m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=12)  # Centered around San Francisco

    # Plot the sensors on the map
    for node_index in range (0, nCount):
        row = traffic_sensors_df[traffic_sensors_df['index'] == node_index]
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=16,
            fill=True,
            opacity=0,
            fill_color='#7FA6EE',
            fill_opacity=0.6,
        ).add_to(m)
        
    # Plot the new sensors on the map
    for node in new_node_list:
        row = traffic_sensors_df[traffic_sensors_df['index'] == node]
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=16,
            fill=True,
            opacity=0,
            fill_color='red',
            fill_opacity=0.9,
        ).add_to(m)
    
    # Plot the removed sensors on the map
    for node in removed_node_list:
        row = traffic_sensors_df[traffic_sensors_df['index'] == node]
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=16,
            fill=True,
            opacity=0,
            fill_color='blue',
            fill_opacity=0.9,
        ).add_to(m)

    return m

for year in range(2011, 2018):
    graph_path = os.path.join(graph_dir, str(year)+"_adj.npz")
    graph_np = np.load(graph_path)['x']
    graph = nx.from_numpy_matrix(graph_np)
    
    nCount = graph.number_of_nodes()
    eCount = graph.number_of_edges()
    nCount_draw = 0
    
    previous_graph = None
    
    

    remove_nodes = []
    remove_edges = []
    
    new_edges = []
    new_nodes = []
    
    if year == 2011:
        pos = extract_pos(meta_df, nCount)
        nCount_draw = nCount
        
    else:
        new_edges = []
        new_nodes = []
        
        pos_new_node = extract_pos(meta_df, nCount)
        
        for node_key in list(pos_new_node.keys()):
            if pos.get(node_key) is None:
                new_nodes.extend([node_key])
                pos[node_key] = pos_new_node[node_key]
                new_edges.extend(graph.edges([node_key]))
        
        remove_nodes = np.load(f'/home/s223719687/python_project/traffic_evolve/data/PeMS3_Expand/FastData_removed/{year}_remove_id.npz')['id']
        remove_edges.extend(graph.edges(remove_edges))
        
        nCount_draw = prev_nCount
    
    map_output = generate_sensor_map(meta_df, nCount_draw, new_nodes, remove_nodes)
    
    img_data = map_output._to_png(3)
    
    img = Image.open(io.BytesIO(img_data))
    img.save(f'fig/pems03/test3/{year}.png')
    
    prev_nCount = nCount