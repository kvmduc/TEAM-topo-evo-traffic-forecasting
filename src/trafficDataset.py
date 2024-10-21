import numpy as np
import torch 
from torch_geometric.data import Data, Dataset

class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] # [T, Len, N]
            self.y = inputs[split+'_y'] # [T, Len, N]
        else:
            self.x = x
            self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 0)
        y = torch.Tensor(self.y[index].T)
        y = torch.unsqueeze(y, 0)

        # x (b,N,F,T)
        # y (b,N,T)
        return Data(x=x, y=y)  
    
class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):
        self.x = inputs # [T, Len, N]
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 0)
        # (b,N,F,T)
        return Data(x=x)