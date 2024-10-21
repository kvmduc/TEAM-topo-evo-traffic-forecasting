import random
import numpy as np
from scipy.stats import wasserstein_distance, entropy as kldiv
import networkx as nx
import os.path as osp


def replay_node_selection(args, pre_data=None, cur_data=None, pre_adj=None, cur_adj=None):
    # data (num_data, node)
    if args.replay_strategy == 'random':
        return random_sampling(pre_data.shape[1], args.replay_num_samples)
    elif args.replay_strategy == 'distribution':
        return select_stablest_node(args, pre_data, cur_data)
    elif args.replay_strategy == 'degree':
        return degree_node(pre_adj, args.replay_num_samples)
    elif args.replay_strategy == 'data_all':
        return select_stablest_node_all_data(args, pre_data, cur_data)
    elif args.replay_strategy == 'centrality':
        return central_node(pre_adj, args.replay_num_samples)
    
def select_stablest_node_all_data(args, pre_data, cur_data):
    pre_data = []
    max_columns = cur_data.shape[1]
    for year in range(args.begin_year, args.year):
        data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
        padded_data = np.pad(data, ((0, 0), (0, max_columns - data.shape[1]), (0,0)), mode='constant')
        pre_data.append(padded_data)
    pre_data = np.concatenate(pre_data, axis=0)
    cur_data = cur_data[:-1,:]
    if pre_data.ndim == 3 and cur_data.ndim == 3:
        pre_data = pre_data[:,:,0]
        cur_data = cur_data[:,:,0]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        # score.append(kldiv(pre_prob, cur_prob))
        score.append(wasserstein_distance(pre_prob, cur_prob))
    # return staiton_id of args.replay_num_samples min score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), args.replay_num_samples)[:args.replay_num_samples]

def random_sampling(data_size, num_samples):
    return np.random.choice(data_size, num_samples)

def degree_node(pre_adj, num_samples):
    graph = nx.from_numpy_array(pre_adj)
    # Get the degree of each node
    degrees = dict(graph.degree())
    # Sort nodes by degree in ascending order
    sorted_nodes = sorted(degrees, key=degrees.get)
    # Select the top k nodes
    lowest_degree_nodes = sorted_nodes[:num_samples]
    return lowest_degree_nodes

def central_node(pre_adj, num_samples):
    graph = nx.from_numpy_array(pre_adj)
    # Get the degree of each node
    centrality_score = nx.closeness_centrality(graph)
    sorted_nodes = sorted(centrality_score, key=centrality_score.get)
    # Select the top k nodes
    lowest_degree_nodes = sorted_nodes[:num_samples]
    return lowest_degree_nodes

def select_stablest_node(args, pre_data, cur_data):
    pre_data = pre_data[-288*1-1:-1,:]                  # 1 day
    cur_data = cur_data[0:288*1,:]                      # next 1 day
    if pre_data.ndim == 3 and cur_data.ndim == 3:
        pre_data = pre_data[:,:,0]
        cur_data = cur_data[:,:,0]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        # score.append(kldiv(pre_prob, cur_prob))
        score.append(wasserstein_distance(pre_prob, cur_prob))
    # return staiton_id of args.replay_num_samples min score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), args.replay_num_samples)[:args.replay_num_samples]