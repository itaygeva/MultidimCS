import torch.linalg
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARGS=torch.load('ARGS.pt')

class DECODER:
  def __init__(self):
    self.model_dict=torch.load('model_dict.pt')
    self.A =self.model_dict['A']     #size (m,n)      in DECONET: m=NUM_MEASURMENTS, n=num_sensors*num_points_in_time
    self.phi =self.model_dict['phi']
    self.ys_test =torch.load('ys_test.pt')
    self.ys_test_noisy=self.noisy_measure(self.ys_test)
    self.x0 = torch.einsum("am,bm->ba", self.A.t(), self.ys_test_noisy)
    self.mu=torch.load('mu.pt')
    self.phix0 = torch.einsum("sa,ba->bs", self.phi, self.x0)
    self.first_activation = TruncationActivation()
    self.second_activation = ShrinkageActivation()
    self.acf_iterations=ARGS.layers
    self.alpha=ARGS.a
    self.beta=ARGS.b


  def initial(self):
    [min_x,max_x]=torch.load('min_max_x')


    z1 = torch.zeros_like(self.phix0)
    z2 = torch.zeros_like(self.ys_test)

    epsilon = torch.norm(self.ys_test - self.ys_test_noisy)

    x_hat = self.decode(self.ys_test_noisy, epsilon, self.mu, self.x0, z1, z2, min_x, max_x)

    return x_hat

  def decode(self, y, epsilon, mu, x0, z1, z2, min_x, max_x):
    u1 = z1
    u2 = z2
    t1 = ARGS.t10
    t2 = ARGS.t20
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

  def affine_transform1(self, theta1, u1, z1, t1, x):
    affine1 = (
            (1 - theta1) * u1
            + theta1 * z1
            - (t1 / theta1) * torch.einsum("sa,ba->bs", self.phi, x)
    )

    return affine1.detach()

  def affine_transform2(self, theta2, u2, z2, t2, y, x):
    affine2 = (
            (1 - theta2) * u2
            + theta2 * z2
            - (t2 / theta2) * (y - torch.einsum("sa,ba->bs", self.A, x))
    )

    return affine2.detach()

class ShrinkageActivation(nn.Module):
    def __init__(self):
      super(ShrinkageActivation, self).__init__()

    def forward(self, x, epsilon):
      return torch.sign(x) * torch.max(torch.zeros_like(x), torch.abs(x) - epsilon)




class TruncationActivation(nn.Module):
  def __init__(self):
    super(TruncationActivation, self).__init__()

  def forward(self, x, epsilon):
    return torch.sign(x) * torch.min(torch.abs(x), epsilon * torch.ones_like(x))






if __name__ == "__main__":
  d=DECODER()
  x_hat=d.initial()




