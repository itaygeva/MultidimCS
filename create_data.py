
import numpy as np

class Data:
    def create_X(self,rows,cols, X_c_energy = 0.7, noise_energy=1e-6): #rows=400, cols=13, rows*cols
        """ x_400*13 = X_c + X_inv + epsilon
        X_c_energy:  the percentage of energy that the common holds. (0.7 for example..)
        """
        full_energy = 1
        lambda1= full_energy * X_c_energy
        u1= np.random.randn(rows,1)
        v1=np.random.randn(1,cols)
        X_c = lambda1 * np.dot(u1,v1)

        X_inv =self.create_inovation(rows,cols)
        epsilon = np.random.randn(rows,cols) * full_energy * noise_energy

        X = X_c + X_inv +epsilon
        return X

    def create_inovation(self,rows,cols):
        t= np.arange(0,rows)/ rows
        angels_deg = np.array([0,30,60]).reshape(1,-1) #up to us.. cos(alp1*t) + cos(alp2*t) + ...
        angels_rad = np.deg2rad(angels_deg)

        cos_function = np.cos(np.sum(t.reshape(-1,1) * angels_rad, axis=1))
        fft_cos = np.fft.fft(cos_function).reshape(-1,1)

        alphas = np.random.randn(1,cols) * 10 #10? whatever we want
        X_inv = fft_cos * alphas

        return X_inv




# data=Data()
# X = data.create_data(400,13)
# print('bla')

# a = 8 + np.random.randn(400,13)
# U,S,Vh = np.linalg.svd(a,full_matrices=False)
# print(U.shape, S.shape, Vh.shape)
# smat = np.diag(S)
# sv=np.dot(smat,Vh)
# print('bla')