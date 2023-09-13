import torch.linalg
import encoder

class DECODER:
  def __init__(self):
    self.A =torch.load('A.pt')     #size (m,n)      in DECONET: m=NUM_MEASURMENTS, n=num_sensors*num_points_in_time
    self.ys_test =torch.load('ys_test.pt')
    self.x0 = torch.einsum("am,bm->ba", self.A.t(), self.ys_test)
    self.phi = torch.load('phi.pt')
    self.mu=torch.load('mu.pt')
    self.theta1=torch.load('theta1.pt')
    self.theta2 = torch.load('theta2.pt')
  def load_and_decode(self):
    self.x_hat = (
            self.x0
            + (
                    (1 - self.theta1) * torch.einsum("as,bs->ba", self.phi.t(), u1)
                    + self.theta1 * torch.einsum("as,bs->ba", self.phi.t(), z1)
                    - (1 - self.theta2) * torch.einsum("am,bm->ba", self.A.t(), u2)
                    - self.theta2 * torch.einsum("am,bm->ba", self.A.t(), u2)
            )
            / self.mu
    )



if __name__ == "__main__":
  d=DECODER()


