
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
    def __init__(self,rows,cols,sig1vt_supp,k_sparse,zt_noise_sigma):
        self.rows=rows
        self.cols=cols
        self.sig1vt_supp=sig1vt_supp
        self.k_sparse=k_sparse   #how many non zero elements, in precentage from the full matrix
        self.num_zeros=self.rows*self.cols*(100-self.k_sparse)   #how many zero elements
        self.zt_noise_sigma=zt_noise_sigma

    def create_X_C_Z(self,save_tag=False): #rows=400, cols=13, rows*cols
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
        if save_tag==False:
            return X,C,Z
        else: 
            return X,C,Z,self.Zt


    def create_Ct(self):
        ut=generate_brownian_noise(duration=1, fs=self.rows, center=1) #ut = u tag:)
        sig1vt=np.random.uniform(-self.sig1vt_supp, self.sig1vt_supp, self.cols)
        Ct=np.reshape(ut,(ut.shape[0],1)) @ np.reshape(sig1vt, (1,sig1vt.shape[0]))
        return Ct


    def create_Zt(self):
        mu=0
        sigma=self.zt_noise_sigma #it was 0.01 in thr beginning
        num_non_zeros=round(self.k_sparse*(self.cols*self.rows)*0.01)
        k_col_rand=np.random.choice(self.cols, num_non_zeros, replace=True)
        k_row_rand = np.random.choice(self.rows, num_non_zeros, replace=True)
        matrix = np.zeros((self.rows,self.cols))
        matrix[k_row_rand,k_col_rand]=1
        matrix=matrix+np.random.normal(mu, sigma, (self.rows, self.cols))
        Zt=matrix
        return Zt

    def create_Dataset(self,num,save_tag=False):
        X_dataset=np.zeros((num,self.rows,self.cols))
        C_dataset = np.zeros((num, self.rows, self.cols))
        Z_dataset = np.zeros((num, self.rows, self.cols))
        for i in range(num):
            X,C,Z=self.create_X_C_Z(save_tag=save_tag)
            X_dataset[i,:,:] = X
            C_dataset[i, :, :] = C
            Z_dataset[i, :, :] = Z
        return X_dataset,C_dataset,Z_dataset
    
    def create_Dataset_save_tag(self,num,save_tag=True):
        X_dataset=np.zeros((num,self.rows,self.cols))
        C_dataset = np.zeros((num, self.rows, self.cols))
        Z_dataset = np.zeros((num, self.rows, self.cols))
        Zt_dataset = np.zeros((num, self.rows, self.cols))
        for i in range(num):
            X,C,Z,Zt=self.create_X_C_Z(save_tag=save_tag)
            X_dataset[i,:,:] = X
            C_dataset[i, :, :] = C
            Z_dataset[i, :, :] = Z
            Zt_dataset[i, :, :] = Zt
        return X_dataset,C_dataset,Z_dataset,Zt_dataset



if __name__ == "__main__":
    pass
    # data=Data(rows=10,cols=10,sig1vt_supp=10,k_sparse=50,zt_noise_sigma=0.01)
    # X,C,Z=data.create_X_C_Z()
    # print(Z)