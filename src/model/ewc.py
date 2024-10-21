import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import numpy as np
import logging
import pdb



class EWC(nn.Module):

    def __init__(self, model, adj, ewc_lambda = 0, ewc_type = 'ewc'):
        super(EWC, self).__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type
        self.adj = adj

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, loader, lossfunc, device):
        # _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        _buff_param_names = []
        _buff_param_data_shape = []
        for param in self.model.named_parameters():
            _buff_param_names.append(param[0].replace('.', '__'))
            _buff_param_data_shape.append(param[1].shape)
        est_fisher_info = {name: 0.0 for name in _buff_param_names}
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = self.model.forward(inputs, self.adj)
            log_likelihood = lossfunc(labels, pred, 0.0)
            grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters(), allow_unused = True)
            for name, weight_shape, grad in zip(_buff_param_names, _buff_param_data_shape, grad_log_liklihood):
                if grad is None:
                    est_fisher_info[name] = torch.zeros(weight_shape).to(device)
                    continue
                est_fisher_info[name] += grad.data.clone() ** 2
        for name in _buff_param_names:
            # print(name, est_fisher_info[name])
            self.register_buffer(name + '_estimated_fisher', est_fisher_info[name])


    def register_ewc_params(self, loader, lossfunc, device):
        self._update_fisher_params(loader, lossfunc, device)
        self._update_mean_params()


    def compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            if estimated_fisher == None:
                losses.append(0)
            elif self.ewc_type == 'l2':
                losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
            else:
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        return 1 * (self.ewc_lambda / 2) * sum(losses)
    
    def forward(self, data, adj): 
        return self.model(data, adj)

