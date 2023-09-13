import torch.linalg
import correlated_noise_dataset as cnd
import numpy as np

class ENCODER:
  def __init__(self, A,xs_test):
    self.A = A      #size (m,n)      in DECONET: m=NUM_MEASURMENTS, n=num_sensors*num_points_in_time
    self.xs_test=xs_test #size (n,p)      n=num_sensors*num_points_in_time, p=num_records
    #self.ys_test=list(map(self.noisy_measure(self.measure_x),list(self.xs_test.t())))
    self.ys_test =self.noisy_measure(self.measure_x(self.xs_test.t()))

  def measure_x(self, x):  #x suppused to enter the function as a row vec.
    # Create measurements y
    y = torch.einsum("ma,ba->bm", self.A, x)
    return y

  def noisy_measure(self, y):
    # add Gaussian noise to y
    y_noisy = y + 0.0001 * torch.randn_like(y)
    return y_noisy


if __name__ == "__main__":
  A=torch.load('A.pt')
  xs_test=cnd.synthetic_generator(100, 3, 1000) #1000 records in test * (3 senseors*100 points in time)
  enc=ENCODER(A,xs_test)
  torch.save(enc.ys_test,'ys_test.pt')


