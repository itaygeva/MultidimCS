
import numpy as np


def generate_brownian_noise(duration=1, fs=400, center=1):
    """
    Generate Brownian noise (red noise) centered around a specified value.

    :param duration: Length of the noise signal in seconds.
    :param fs: Sampling frequency in Hz.
    :param center: The center value around which the noise oscillates.
    :return: NumPy array containing Brownian noise centered at the given value.
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

    def create_Ct(self):
        ut=generate_brownian_noise(duration=1, fs=400, center=1) #ut = u tag:)
        sig1vt=np.random.uniform(-10, 10, 13)



    def create_Zt(self,rows,cols):
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