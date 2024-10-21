import sys, json, argparse, random, re, os, shutil
sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb
from utils.data_convert import generate_samples


from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp


from torch_geometric.utils import k_hop_subgraph
from lib.utils import cheb_polynomial, scaled_Laplacian, cheb_polynomial_cos

from utils import common_tools as ct
from lib.metrics import masked_mae, masked_rmse, masked_mse
from lib.utils import load_custom_graphdata, predict_and_save_results_mstgcn
from src.model.model import make_model
from src.model.ewc import EWC

from src.model import detect
from src.model import replay
from tqdm import tqdm

result = {3:{"mae":{}, "mape":{}, "rmse":{}}, 6:{"mae":{}, "mape":{}, "rmse":{}}, 12:{"mae":{}, "mape":{}, "rmse":{}}}
pin_memory = False
n_work = 0 

def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def load_best_model(args):
    epo_list = []
    for filename in os.listdir(osp.join(args.model_path, args.logname+args.time, str(args.year-1))): 
        if filename.endswith(".tar"):
            epo_list.append(filename[6:]) 					
    epo_list= sorted(epo_list)
    params_path_prev_year = osp.join(args.model_path, args.logname+args.time, str(args.year-1))
    load_path = '{}/epoch_{epo_num}'.format(params_path_prev_year, epo_num = epo_list[-1])
    assert os.path.exists(load_path), 'Weights at {} not found'.format(load_path)

    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path)["model_state_dict"]
    assert os.path.exists(load_path), 'Weights at {} not found'.format(load_path)

    model = make_model(DEVICE = args.device, nb_block = args.nb_block, in_channels = args.in_channels, K = args.K, nb_chev_filter = args.nb_chev_filter, nb_time_filter = args.nb_time_filter, time_strides = args.num_of_hours, num_for_predict = args.y_len, len_input = args.x_len)
    model_dict = model.state_dict()
    state_dict = { k:v for k,v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size() }

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    args.logger.info('load weight from: {}'.format(load_path))
    return model,0

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H_%M_%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    
    ct.mkdirs(vars(args)["path"])
    del info


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(train_x_tensor, train_target_tensor, val_x_tensor, val_target_tensor, test_x_tensor, test_target_tensor, args, stablest_nodes=None, influence_nodes=None):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)


    if args.loss=='masked_mse':
        lossfunc = masked_mse         #nn.MSELoss().to(DEVICE)
    elif args.loss=='masked_mae':
        lossfunc = masked_mae
    elif args.loss=='huber':
        lossfunc = torch.nn.functional.huber_loss

    # Dataset Definition
    if args.strategy == 'incremental' and args.year > args.begin_year:

        # ------- train_loader -------
        train_dataset = torch.utils.data.TensorDataset(train_x_tensor[:, args.subgraph.numpy(), :, :], train_target_tensor[:, args.subgraph.numpy(), :])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x_tensor[:, args.subgraph.numpy(), :, :], val_target_tensor[:, args.subgraph.numpy(), :])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        vars(args)["sub_adj"] = adj
    else:
        # ------- train_loader -------
        train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # ------- val_loader -------
        val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        vars(args)["sub_adj"] = vars(args)["adj"]
    # ------- test_loader -------
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # ------- prepare_filter -------
    vars(args)["sub_L_tilde"] = scaled_Laplacian(args.sub_adj)
    vars(args)["sub_cheb_polynomials"] = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in cheb_polynomial(args.sub_L_tilde, args.K)] 


    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args) 
        if args.ewc:
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))
            model = EWC(gnn_model, args.cheb_polynomials, args.ewc_lambda, args.ewc_strategy)
            ewc_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
            ewc_loader = torch.utils.data.DataLoader(ewc_dataset, batch_size=args.batch_size, shuffle=True)
            model.register_ewc_params(ewc_loader, lossfunc, device)
        else:
            model = gnn_model
    else:
        model = make_model(DEVICE = args.device, nb_block = args.nb_block, in_channels = args.in_channels, K = args.K, nb_chev_filter = args.nb_chev_filter, nb_time_filter = args.nb_time_filter, time_strides = args.num_of_hours, num_for_predict = args.y_len, len_input = args.x_len)
    
    # Model Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    # global_train_steps = len(train_loader) // args.batch_size +1

    # iters = len(train_loader)
    lowest_validation_loss = np.inf
    wait = 0
    patience = 100
    best_epoch = 0

    use_time = []
    for epoch in range(args.epoch):

        params_filename = os.path.join(args.model_path, args.logname+args.time, str(args.year),'epoch_%s.tar' % epoch)
        cn = 0
        training_loss = 0.0
        validation_loss = 0


        start_time = datetime.now()

        # Train Model
        model.train()
        
        for batch_idx, data in tqdm(enumerate(train_loader), total = len(train_loader)):
            
            inputs, labels = data              # (b,N,F,T) & (b,N,T)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(inputs.shape))
            # data = data.to(device, non_blocking=pin_memory)


            optimizer.zero_grad()

            outputs = model(inputs, args.sub_cheb_polynomials)

            if args.strategy == "incremental" and args.year > args.begin_year:
                # pred, _ = to_dense_batch(pred, batch=data.batch)                      # unblock if batchsize < new node number
                # data.y, _ = to_dense_batch(data.y, batch=data.batch)                  # unblock if batchsize < new node number
                outputs = outputs[:, args.mapping, :]
                labels = labels[:, args.mapping, :]

            loss = lossfunc(outputs, labels, 0.0)

            training_loss = loss.item()

            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()

            loss.backward()
            optimizer.step()
            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn 
 
        # Validate Model
        model.train(False)  # ensure dropout layers are in evaluation mode
        with torch.no_grad():
            cn = 0

            for batch_idx, data in enumerate(val_loader):
                inputs, labels = data              # (b,N,F,T) & (b,N,T)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                # data = data.to(device,non_blocking=pin_memory)
                outputs = model(inputs, args.sub_cheb_polynomials)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    outputs = outputs[:, args.mapping, :]
                    labels = labels[:, args.mapping, :]
                loss = lossfunc(outputs, labels, 0.0)
                validation_loss += loss.item()
                cn += 1
        validation_loss = float(validation_loss)/cn             # average validation loss

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.2f} validation loss:{validation_loss:.2f}")

        if validation_loss < lowest_validation_loss:
            wait = 0
            lowest_validation_loss = validation_loss
            best_epoch = epoch
            if not os.path.exists(str(args.model_path) + '/' + str(args.logname) + str(args.time) + '/' +  str(args.year) + '/'):
                os.makedirs(str(args.model_path) + '/' + str(args.logname) + str(args.time) + '/' +  str(args.year) + '/')
            torch.save({'model_state_dict': model.state_dict()}, params_filename)
            # save_model
            print('save parameters to file: %s' % params_filename)
        elif validation_loss >= lowest_validation_loss:
            wait += 1
            if wait == patience:
                args.logger.warning('Early stopping at epoch: %d' % epoch)
                break
    
    best_params_filename = os.path.join(args.model_path, args.logname+args.time, str(args.year), 'epoch_%s.tar' % best_epoch)
    model.load_state_dict(torch.load(best_params_filename)["model_state_dict"])

    
    # Test Model

    predict_and_save_results_mstgcn(model, test_loader, test_target_tensor, 'LOREN_IPSUM', 'mask', args.model_path, 'test', args.year, result, args.logger, args.cheb_polynomials, stablest_nodes=stablest_nodes, influence_nodes=influence_nodes)
    result[f"week_{args.year}"] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_params_filename))



def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    for year in range(args.begin_year, args.end_year+1):
        # Load Data 
        graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year

        train_x_tensor, train_target_tensor, val_x_tensor, val_target_tensor, test_x_tensor, test_target_tensor = load_custom_graphdata(args.save_data_path, str(year), args.device)


        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year)))) 

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]

        vars(args)["adj"] = adj
        vars(args)["L_tilde"] = scaled_Laplacian(args.adj)
        vars(args)["cheb_polynomials"] = [torch.from_numpy(i).type(torch.FloatTensor).to(args.device) for i in cheb_polynomial(args.L_tilde, args.K)]

        replay_node_list = None
        influence_node_list = None
        if args.strategy == "retrain" and year > args.begin_year:
            # Load the best model
            model, _ = load_best_model(args)

            cur_adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
            pre_adj = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]
            
            pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
            cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
            
            args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
            pre_graph = np.array(list(nx.from_numpy_matrix(pre_adj).edges)).T
            cur_graph = np.array(list(nx.from_numpy_matrix(cur_adj).edges)).T
            ################ Influence nodes ################
            args.logger.info("Top k ratio {}".format(args.topk_ratio))
            vars(args)["topk"] = int(args.topk_ratio * args.graph_size) 
            influence_node_list = detect.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
            print(f"Influence nodes: {influence_node_list}")
            
            ################ Replay nodes ################
            vars(args)["replay_num_samples"] = int(args.topk_ratio * args.graph_size)
            args.logger.info("[*] replay node number {}".format(args.replay_num_samples))
            replay_node_list = replay.replay_node_selection(args, pre_data, cur_data, pre_adj, cur_adj)
            print(f"Replay nodes: {replay_node_list}")
        
        if year > args.begin_year and args.strategy == "incremental":
                
            # Load the best model
            model, _ = load_best_model(args)
            
            node_list = list()
            
            cur_adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
            pre_adj = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"]
            
            pre_data = np.load(osp.join(args.raw_data_path, str(year-1)+".npz"))["x"]
            cur_data = np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"]
            
            # Obtain increase nodes
            if args.increase:
                cur_node_size = cur_adj.shape[0]
                pre_node_size = pre_adj.shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))

            print(f"Number of increased nodes: {len(node_list)}")
            
            # Obtain remove nodes
            if args.decrease:
                remove_id = np.load(osp.join(args.save_data_path, f'{year}_remove_id.npz'))['id'].tolist()
                print(f"Number of removed nodes: {len(remove_id)}")
                node_list.extend(remove_id)
            
            
                                
            # Obtain influence nodes
            if args.detect:
                args.logger.info("[*] detect strategy {}".format(args.detect_strategy))
                pre_graph = np.array(list(nx.from_numpy_matrix(pre_adj).edges)).T
                cur_graph = np.array(list(nx.from_numpy_matrix(cur_adj).edges)).T
                ################ CHECK top k ratio ################
                args.logger.info("Top k ratio {}".format(args.topk_ratio))
                vars(args)["topk"] = int(args.topk_ratio * args.graph_size) 
                influence_node_list = detect.influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph)
                print(f"Number of influence nodes: {len(influence_node_list)}")
                node_list.extend(list(influence_node_list))

            # Obtain sample nodes
            if args.replay:
                vars(args)["replay_num_samples"] = int(args.topk_ratio * args.graph_size)
                args.logger.info("[*] replay node number {}".format(args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(args, pre_data, cur_data, pre_adj, cur_adj)
                node_list.extend(list(replay_node_list))
            
            node_list = list(set(node_list))
            
            # Obtain subgraph of node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
            
            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)


            # Skip the year when no nodes needed to be trained incrementally
            if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
                model, loss = load_best_model(args)
                ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
                torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".tar"))
                test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.batch_size, shuffle=False)
                predict_and_save_results_mstgcn(model, test_loader, test_target_tensor, 'LOREN_IPSUM', 'mask', args.model_path, 'test', args.year, result, args.logger)
                logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
                continue

        if args.train:
            train(train_x_tensor, train_target_tensor, val_x_tensor, val_target_tensor, test_x_tensor, test_target_tensor, args, stablest_nodes=None, influence_nodes=None)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.batch_size, shuffle=False)
                predict_and_save_results_mstgcn(model, test_loader, test_target_tensor, 'LOREN_IPSUM', 'mask', args.model_path, 'test', args.year, result, args.logger)


    for i in [3, 6, 12]:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            for year in range(args.begin_year, args.end_year+1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            info+="{:.2f}\t".format(result[i][j][year])
            logger.info("{}\t{}\t".format(i,j) + info)

    for year in range(args.begin_year, args.end_year+1):
        if f"week_{year}" in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[f"week_{year}"]["total_time"], result[f"week_{year}"]["average_time"], result[f"week_{year}"]['epoch_num'])
            logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "conf/trafficStream_pems04.json")
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--topk_ratio", type = float, default = 0.15)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "res/district3F11T17/TrafficStream2021-05-09-11:56:33.516033/2011/27.4437.tar", help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    seed_set(13)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    
    main(args)