
from datetime import datetime
from numpy import argmin, linalg as la
from scipy.linalg import block_diag
import torch
from torch.linalg import matrix_norm as mn
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam, LBFGS
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # type: ignore
from torchvision.transforms import Compose, Normalize, ToTensor  # type: ignore
import matplotlib.pyplot as plt
import create_data_upd as OD


AMBIENT_DIM=400
N_SENSORS=10
    
len_train=7000; len_val=1000
rows=AMBIENT_DIM; cols=N_SENSORS; k_sparse=5; sig1vt_supp=10
# data = OD.Data(rows=rows, cols=cols, sig1vt_supp=sig1vt_supp, k_sparse=k_sparse)
# X_dataset_train, C_dataset_train, Z_dataset_train = data.create_Dataset(len_train)
# X_dataset_val, C_dataset_val, Z_dataset_val = data.create_Dataset(len_val)

# tensors={
#     'X_dataset_train':X_dataset_train,
#     'C_dataset_train':C_dataset_train,
#     'Z_dataset_train':Z_dataset_train,
#     'X_dataset_val':X_dataset_val,
#     'C_dataset_val':C_dataset_val,
#     'Z_dataset_val':Z_dataset_val,
# }
# print('saving...')
checkpoint_name = f"save_datasets_{rows}x{cols}_ksparse={k_sparse}%_sig1vt_supp={sig1vt_supp}.pt"
#torch.save(tensors, checkpoint_name)

loaded_tensors=torch.load(checkpoint_name)
print('bla')

# Z_dataset_train_vecs=Z_dataset_train.transpose(2, 0, 1).reshape(cols,rows*len_train).transpose()
# Z_dataset_val_vecs=Z_dataset_val.transpose(2, 0, 1).reshape(cols, rows * len_val).transpose()