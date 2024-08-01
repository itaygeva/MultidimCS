import torch
from torch.linalg import matrix_norm as mn
import numpy as np
import create_data
from PIL import Image


ambient = 28*28  #784, MNIST
measurements=round(0.25 * 784)
sparse_dim=10
mu = 100 #deafult

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class ENCODER:
    def __init__(self,A):
        self.A = A # size: (m ,amb), m=28*28*0.25, amb=28*28

    def measure_x(self, x):
        # Create measurements y
        y = torch.einsum("ma,ba->bm", self.A, x)
        return y

    def forward(self, x):
        x = x.view(x.size(0), -1)
        min_x = torch.min(x)
        max_x = torch.max(x)

        y = self.measure_x(x)
        return y,min_x,max_x



""" sending in the network: A, min_x,max_x,y """

class Decoder:
    def __init__(self, A,y,mu, phi,min_x, max_x):
        self.A = A
        self.y = y
        self.min_x = min_x
        self.max_x = max_x

        self.y_noisy = self.noisy_measure(y)
        self.epsilon = torch.norm(self.y - self.y_noisy)
        self.mu = mu
        self.x0 =  torch.einsum("am,bm->ba", self.A.t(), self.y_noisy)

        self.sparse_dim = 7840 ##for example for MNIST
        self.ambient = 28*28 ##for example for MNIST

        """learned parameter"""
        self.phi = phi

    def decode(self, y, epsilon, mu, x0, z1, z2, min_x, max_x):
        u1 = z1
        u2 = z2
      #  t1 = ARGS.t10
       # t2 = ARGS.t20
        t1=1
        t2=1 #deafult

        theta1 = 1
        theta2 = 1
        Lexact = torch.tensor([1000.0]).to(DEVICE)

        for _ in range(self.acf_iterations):
            x_hat = (
                    x0
                    + (
                            (1 - theta1) * torch.einsum("as,bs->ba", self.phi.t(), u1)
                            + theta1 * torch.einsum("as,bs->ba", self.phi.t(), z1)
                            - (1 - theta2) * torch.einsum("am,bm->ba", self.A.t(), u2)
                            - theta2 * torch.einsum("am,bm->ba", self.A.t(), u2)
                    )
                    / mu
            )

            w1 = self.affine_transform1(theta1, u1, z1, t1, x_hat)
            w2 = self.affine_transform2(theta2, u2, z2, t2, y, x_hat)

            z1 = self.first_activation(w1, t1 / theta1)
            z2 = self.second_activation(w2, t2 * epsilon / theta2)
            u1 = (1 - theta1) * u1 + theta1 * z1
            u2 = (1 - theta2) * u2 + theta2 * z2

            t1 = self.alpha * t1
            t2 = self.beta * t2
            muL = torch.sqrt(mu / Lexact).to(DEVICE)
            theta_scale = (1 - muL) / (1 + muL).to(DEVICE)
            theta1 = torch.min(torch.tensor([1.0]).to(DEVICE), theta1 * theta_scale)
            theta2 = torch.min(torch.tensor([1.0]).to(DEVICE), theta2 * theta_scale)

        return torch.clamp(x_hat, min=min_x, max=max_x)


    def noisy_measure(self, y):
        # add Gaussian noise to y
        y_noisy = y + 0.0001 * torch.randn_like(y)
        return y_noisy

    def forward(self, y):
        y_noisy = self.noisy_measure(y)

        x0 = torch.einsum("am,bm->ba", self.A.t(), y_noisy)
        phix0 = torch.einsum("sa,ba->bs", self.phi, x0)
        mu = self.mu

        z1 = torch.zeros_like(phix0)
        z2 = torch.zeros_like(y)

        epsilon = torch.norm(y - y_noisy)

        x_hat = self.decode(y_noisy, epsilon, mu, x0, z1, z2, self.min_x, self.max_x)

        return x_hat



if __name__ == "__main__":
    ##MNIST
    file_name = 'DECONET(MNIST)-10L-red7840-lr0.01-mu100-initkaiming.pt'
    state_dict = torch.load(file_name)
    A=state_dict['A']
    enc = ENCODER(A)
    img = Image.open('mnist_img_norm.png')
    image_array = np.array(img)
    img_tens = torch.tensor(image_array)
    y, min_x, max_x, A = enc.forward(img_tens)

    phi = state_dict['phi']
    dec = Decoder(A=A, y=y, mu=mu, phi=phi, min_x=min_x, max_x=max_x)
    img_hat = dec.forward(y)


    ##OUR DATA
    # d=create_data.Data()
    # xs_test = d.create_X(400,13,0.7,1e-6)  #xs_test: size = (400,13) , size of the mini matrix.
    #
    # enc = ENCODER()
    # y,min_x,max_x,A = enc.forward(xs_test)
    #
    # phi = torch.empty(sparse_dim, ambient) #we need the learned phi!
    #
    # dec = Decoder(A=A,y=y, mu=mu, phi=phi,min_x=min_x, max_x=max_x)
    # x_hat = dec.forward(y)



