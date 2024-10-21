import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import wasserstein_distance, entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch 
from src.trafficDataset import continue_learning_Dataset
from lib.utils import cheb_polynomial, scaled_Laplacian
from utils.data_convert import generate_dataset
import torch.utils.data
import torch
from scipy.spatial import distance
import os.path as osp
import networkx as nx


def get_feature(data, args, model, adj, year):
    L_tilde = scaled_Laplacian(adj)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in cheb_polynomial(L_tilde, args.K)]
    
    idx = [i for i in range(int(data.shape[0]))]
    x, _ = generate_dataset(data, idx)
    x = np.expand_dims(x, axis = 3)                # (num_data, seq_len, num_node, 1)
    x = np.transpose(x, axes= (0, 2, 3, 1))         # (num_data, num_node, feature, seq_len)

    data = torch.from_numpy(x).type(torch.FloatTensor).to(args.device)
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    features = []
    for batch in dataloader:
        data = batch[0]
        feature = model.feature(data, cheb_polynomials) 
        feature = feature.reshape(feature.shape[0], feature.shape[1], -1)
        feature = feature.permute(1,0,2)
        feature = feature.cpu().detach().numpy()
        features.append(feature)            
    features = np.concatenate(features, axis=1)

    return features


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    return adj
    

def score_func(pre_data, cur_data, args):
    # shape: [T, N]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    # return staiton_id of topk max score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    # detect_strategy: "original": hist of original series; "feature": hist of feature at each dimension
    if args.detect_strategy == 'original':
        pre_data = pre_data[-288*1-1:-1,:]              # 1 day
        cur_data = cur_data[0:288*1,:]                  # next 1 day
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
            score.append(distance.jensenshannon(pre_prob, cur_prob))
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    elif args.detect_strategy == 'feature':
        model.eval()
        pre_data = pre_data[-288*-1:-1,:]
        cur_data = cur_data[-288*-1:-1,:]
        pre_adj = get_adj(args.year-1, args)
        cur_adj = get_adj(args.year, args)
        
        pre_data = get_feature(pre_data, args, model, pre_adj, args.year-1)
        cur_data = get_feature(cur_data, args, model, cur_adj, args.year)

        score = []
        for i in range(pre_data.shape[0]):
            score_ = 0.0
            j = pre_data.shape[2]
            pre_data[i,:,:j] = (pre_data[i,:,:j] - np.min(pre_data[i,:,:j], axis=1, keepdims=True))/(np.max(pre_data[i,:,:j], axis=1, keepdims=True) - np.min(pre_data[i,:,:j], axis=1, keepdims=True))
            cur_data[i,:,:j] = (cur_data[i,:,:j] - np.min(cur_data[i,:,:j], axis=1, keepdims=True))/(np.max(cur_data[i,:,:j], axis=1, keepdims=True) - np.min(cur_data[i,:,:j], axis=1, keepdims=True))
            pre_prob, _ = np.histogram(pre_data[i,:,:j], bins=10, range=(0, 1))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[i,:,:j], bins=10, range=(0, 1))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            # score_ += distance.jensenshannon(pre_prob, cur_prob)
            score_ += wasserstein_distance(pre_prob, cur_prob)
            score.append(score_)
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    
    elif args.detect_strategy == 'degree':
        pre_adj = np.load(osp.join(args.graph_path, str(args.year-1)+"_adj.npz"))["x"]
        graph = nx.from_numpy_array(pre_adj)
        # Get the degree of each node
        degrees = dict(graph.degree())
        # Sort nodes by degree in ascending order
        sorted_nodes = sorted(degrees, key=degrees.get)
        # Select the top k nodes
        highest_degree_nodes = sorted_nodes[-args.topk:]
        return highest_degree_nodes

    elif args.detect_strategy == 'random':
        pre_adj = np.load(osp.join(args.graph_path, str(args.year-1)+"_adj.npz"))["x"]
        return np.random.choice(pre_adj.shape[0], args.topk)
    
    elif args.detect_strategy == 'data_all':
        return select_unstablest_node_all_data(args, pre_data, cur_data)
    
    elif args.detect_strategy == 'centrality':
        pre_adj = np.load(osp.join(args.graph_path, str(args.year-1)+"_adj.npz"))["x"]
        graph = nx.from_numpy_array(pre_adj)
        # Get the degree of each node
        centrality_score = nx.closeness_centrality(graph)
        sorted_nodes = sorted(centrality_score, key=centrality_score.get)
        # Select the top k nodes
        largest_degree_nodes = sorted_nodes[-args.topk:]
        return largest_degree_nodes
        
    else: 
        args.logger.info("node selection mode illegal!")


def select_unstablest_node_all_data(args, pre_data, cur_data):
    pre_data = []
    max_columns = cur_data.shape[1]
    print(max_columns)
    for year in range(args.begin_year, args.year):
        data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
        print(f"data shape: {data.shape}")
        padded_data = np.pad(data, ((0, 0), (0, max_columns - data.shape[1]), (0, 0)), mode='constant')
        print(f"padded_data shape: {padded_data.shape}")
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
        score.append(kldiv(pre_prob, cur_prob))
    # return staiton_id of args.replay_num_samples min score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]