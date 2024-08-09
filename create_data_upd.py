
import numpy as np
import torch
import torch.nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Synthetic(Dataset):
    def __init__(self, input_vec) -> None:

        self.input_vec = input_vec.t()

    def __len__(self):

        return self.input_vec.shape[0]

    def __getitem__(self, index):
        x = self.input_vec[index]

        return x


def generate_brownian_noise(duration=1, fs=400, center=1): #sec,Hz,mean of BN
    """
    Generate Brownian noise (red noise) centered around a specified value.
    we want our vector u to be with 400 elements (like the real Rafael's data), so dur*fs=400
    """
    # Generate white noise
    white_noise = np.random.normal(size=int(duration * fs))
    # Integrate the white noise to create Brownian noise
    brownian_noise = np.cumsum(white_noise)
    brownian_noise = np.interp(brownian_noise, (brownian_noise.min(), brownian_noise.max()),
                               (center - 1, center + 1))
    return brownian_noise

class Data:
    def __init__(self,rows,cols,sig1vt_supp,k_sparse):
        self.rows=rows
        self.cols=cols
        self.sig1vt_supp=sig1vt_supp
        self.k_sparse=k_sparse


    def create_X_C_Z(self): #rows=400, cols=13, rows*cols
        self.Ct=self.create_Ct()
        self.Zt=self.create_Zt()
        X=self.Ct+self.Zt
        U, S, Vt = np.linalg.svd(X, full_matrices=True) #U_rxr, S_1xmin(r,c), Vt_cxc
        min_rc=min(self.rows, self.cols)
        Sigma = np.zeros((U.shape[1], Vt.shape[0]))
        first_s=np.zeros((U.shape[1], Vt.shape[0]))
        np.fill_diagonal(Sigma, S)
        np.fill_diagonal(first_s, np.concatenate((S[0], np.zeros((1,min_rc-1))), axis=None))
        C=U @ first_s @ Vt
        Z=X-C
        return X,C,Z


    def create_Ct(self):
        ut=generate_brownian_noise(duration=1, fs=self.rows, center=1) #ut = u tag:)
        sig1vt=np.random.uniform(-self.sig1vt_supp, self.sig1vt_supp, self.cols)
        Ct=np.reshape(ut,(ut.shape[0],1)) @ np.reshape(sig1vt, (1,sig1vt.shape[0]))
        return Ct


    def create_Zt(self):
        mu=0
        sigma=1
        matrix = np.random.normal(mu, sigma, (self.rows, self.cols))
        k_col_rand=np.random.choice(self.cols, self.k_sparse, replace=True)
        k_row_rand = np.random.choice(self.rows, self.k_sparse, replace=True)
        matrix[k_row_rand,k_col_rand]=0
        Zt=matrix
        return Zt

    def create_Dataset(self,num):
        X_dataset=np.zeros((num,self.rows,self.cols))
        C_dataset = np.zeros((num, self.rows, self.cols))
        Z_dataset = np.zeros((num, self.rows, self.cols))
        for i in range(num):
            X,C,Z=self.create_X_C_Z()
            X_dataset[i,:,:] = X
            C_dataset[i, :, :] = C
            Z_dataset[i, :, :] = Z
        return X_dataset,C_dataset,Z_dataset



# data=Data(rows=400,cols=13,sig1vt_supp=10,k_sparse=200)
# X,C,Z=data.create_X_C_Z()
# print(X,C,Z)