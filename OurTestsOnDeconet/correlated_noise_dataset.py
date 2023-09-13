import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import ToTensor


def generate_highly_correlated_covariance_matrix(num_variables):
    # Create a random covariance matrix with diagonal elements as 1 (variances)
    covariance_matrix = 1 * torch.eye(num_variables)

    # Set off-diagonal elements (covariances) to a positive constant
    constant_covariance = 0.9

    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            covariance_matrix[i, j] = constant_covariance
            covariance_matrix[j, i] = constant_covariance

    return covariance_matrix


class Synthetic(Dataset):
    def __init__(self, input_vec) -> None:
        self.input_vec = input_vec.t()

    def __len__(self):
        return self.input_vec.shape[0]

    def __getitem__(self, index):
        x = self.input_vec[index]

        return x


def synthetic_generator(input_dim, n_sensors, set_size):
    noise = torch.normal(0, 0.01, size=(input_dim, n_sensors, set_size))
    #print('noise',noise)
    cov_matrix = generate_highly_correlated_covariance_matrix(n_sensors)
    cholesky_matrix = torch.linalg.cholesky(cov_matrix)
    #print(cholesky_matrix)
    correlated_noise = torch.matmul(cholesky_matrix, noise)
    input_vectors = correlated_noise.view(input_dim*n_sensors, set_size)

    return input_vectors


# def myfunc(x):
#     return x



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = synthetic_generator(100, 3, 1000)
    print(a.size())
    # P1 = torch.rand(2, 6)
    #
    # print('list P1',list(P1))
    # print('list P1 tag', list(P1.t()))
    # #b=a.view(input_dim*n_sensors, set_size)
    # #ys=map(myfunc,a)
