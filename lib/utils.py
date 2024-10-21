import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from .metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse,masked_mae_test,masked_rmse_test
import logging
import sys
import time

def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def cheb_polynomial_cos(K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cos_cheb_polynomials: np.ndarray size K*K,  cos_cheb_polynomials[j][k] is the order k based on x_j 

    '''
    
    # init a list of x_j
    # x_j = cos((j+1/2)*pi/(K+1))
    
    x = []        # size K

    for j in range(K):
        x.append(np.cos( (j + 1/2) * np.pi / (K+1)))

    # calculate k 

    cos_cheb_polynomials = np.zeros(shape = (K,K))

    for j in range(K):
        
        cos_cheb_polynomials[j, 0] = 1
        cos_cheb_polynomials[j, 1] = x[j]
        
        for i in range (2, K):

            cos_cheb_polynomials[j,i] = 2 * x[j] * cos_cheb_polynomials[j, i - 1] - cos_cheb_polynomials[j,i - 2]


    return cos_cheb_polynomials        # cos_cheb_polynomials[j][k] is the order k of x_j


def load_custom_graphdata(graph_signal_matrix_filename, year, DEVICE):
    '''
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath, str(year)+"_30day.npz")

    print('load file:', filename)

    file_data = np.load(filename, allow_pickle=True)
    

    train_x = file_data['train_x']                              # (num_data, seq_len, num_node)
    train_x = np.expand_dims(train_x, axis = 3)                # (num_data, seq_len, num_node, 1)
    train_x = np.transpose(train_x, axes= (0, 2, 3, 1))         # (num_data, num_node, feature, seq_len)
    train_target = file_data['train_y']                         # (num_data, seq_len, num_node)
    train_target = np.transpose(train_target, axes= (0, 2, 1))  # (num_data, num_node, seq_len)


    val_x = file_data['val_x']                                  # (num_data, seq_len, num_node)
    val_x = np.expand_dims(val_x, axis = 3)                    # (num_data, seq_len, num_node, 1)
    val_x = np.transpose(val_x, axes= (0, 2, 3, 1))             # (num_data, num_node, feature, seq_len)
    val_target = file_data['val_y']                             # (num_data, seq_len, num_node)
    val_target = np.transpose(val_target, axes= (0, 2, 1))      # (num_data, num_node, seq_len)


    test_x = file_data['test_x']                                # (num_data, seq_len, num_node)
    test_x = np.expand_dims(test_x, axis = 3)                  # (num_data, seq_len, num_node, 1)
    test_x = np.transpose(test_x, axes= (0, 2, 3, 1))           # (num_data, num_node, feature, seq_len)
    test_target = file_data['test_y']                           # (num_data, seq_len, num_node)
    test_target = np.transpose(test_target, axes= (0, 2, 1))    # (num_data, num_node, seq_len)


    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)  # (B, N, T)


    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor) # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor)  # (B, N, T)


    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)


    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_x_tensor, train_target_tensor, val_x_tensor, val_target_tensor, test_x_tensor, test_target_tensor



def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value,sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss



def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, params_path, type, year, result, logger, cheb_polynomials, stablest_nodes=None, influence_nodes=None):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data
            device = cheb_polynomials[0].device

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)
            encoder_inputs = encoder_inputs.to(device)
            outputs = net(encoder_inputs, cheb_polynomials)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)

        
        excel_list = []

        for i in [3 , 6, 12]:
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, :, :i], prediction[:, :, :i],0.0)
                rmse = masked_rmse_test(data_target_tensor[:, :, :i], prediction[:, :, :i],0.0)
                mape = masked_mape_np(data_target_tensor[:, :, :i], prediction[:, :, :i], 0)
            else :
                mae = mean_absolute_error(data_target_tensor[:, :, :i], prediction[:, :, :i])
                rmse = mean_squared_error(data_target_tensor[:, :, :i], prediction[:, :, :i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, :, :i], prediction[:, :, :i], 0)
            result[i]['mae'][year] = mae
            result[i]['rmse'][year] = rmse
            result[i]['mape'][year] = mape
            logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        if metric_method == 'mask':
            mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        else :
            mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)
        
        if stablest_nodes is not None:
            logger.info('Evaluate on stablest nodes: {}'.format(len(stablest_nodes)))
            logger.info("Stablest nodes: {}".format(' '.join(map(str, stablest_nodes))))
            print(data_target_tensor[:, stablest_nodes, :i].shape)
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, stablest_nodes, :i], prediction[:, stablest_nodes, :i],0.0)
                rmse = masked_rmse_test(data_target_tensor[:, stablest_nodes, :i], prediction[:, stablest_nodes, :i],0.0)
                mape = masked_mape_np(data_target_tensor[:, stablest_nodes, :i], prediction[:, stablest_nodes, :i], 0)
            else :
                mae = mean_absolute_error(data_target_tensor[:, stablest_nodes, :i], prediction[:, stablest_nodes, :i])
                rmse = mean_squared_error(data_target_tensor[:, stablest_nodes, :i], prediction[:, stablest_nodes, :i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, stablest_nodes, :i], prediction[:, stablest_nodes, :i], 0)

            logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))

        if influence_nodes is not None:
            logger.info('Evaluate on influence nodes: {}'.format(len(influence_nodes)))
            logger.info("Influence nodes: {}".format(' '.join(map(str, influence_nodes))))
            print(data_target_tensor[:, influence_nodes, :i].shape)
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, influence_nodes, :i], prediction[:, influence_nodes, :i],0.0)
                rmse = masked_rmse_test(data_target_tensor[:, influence_nodes, :i], prediction[:, influence_nodes, :i],0.0)
                mape = masked_mape_np(data_target_tensor[:, influence_nodes, :i], prediction[:, influence_nodes, :i], 0)
            else :
                mae = mean_absolute_error(data_target_tensor[:, influence_nodes, :i], prediction[:, influence_nodes, :i])
                rmse = mean_squared_error(data_target_tensor[:, influence_nodes, :i], prediction[:, influence_nodes, :i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, influence_nodes, :i], prediction[:, influence_nodes, :i], 0)

            logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))




def init_log():
    log_dir = './log/'
    log_filename = 'info_%s' % time.strftime('%m-%d-%H-%M-%S')
    logger = logging.getLogger(__name__)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
    # vars(args)["logger"] = logger
    return logger


def load_masked_test_dataset(year, dataset_dir, DEVICE):

    filename = osp.join(dataset_dir, str(year)+"_30day.npz")

    print('load file:', filename)

    file_data = np.load(filename, allow_pickle=True)
    test_x = file_data['test_x']                                # (num_data, seq_len, num_node)
    test_x = np.expand_dims(test_x, axis = 3)                  # (num_data, seq_len, num_node, 1)
    test_x = np.transpose(test_x, axes= (0, 2, 3, 1))           # (num_data, num_node, feature, seq_len)

    test_target = file_data['test_y']                           # (num_data, seq_len, num_node)
    test_target = np.transpose(test_target, axes= (0, 2, 1))    # (num_data, num_node, seq_len)

    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    return test_x_tensor, test_target_tensor


