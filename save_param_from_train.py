import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
#
# file_name='DECONET(MNIST)-10L-red7840-lr0.01-mu100-initkaiming.pt'
# state_dict=torch.load(file_name)
# phi=state_dict['phi']
# A=state_dict['A']


# data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) #to tensor: 0-1
# mnist_data = datasets.MNIST(root='.', train=True, download=False, transform=data_transform)
# data_loader = DataLoader(mnist_data, batch_size=1, shuffle=True)
#
# images, labels = next(iter(data_loader))
# image = images[0].squeeze()  # Remove the batch dimension and get the first image
# pil_image = Image.fromarray((image.numpy()).astype('uint8'), mode='L') #after normalize...
#
# # Save the image to the local file system
# pil_image.save('mnist_img_norm.png')
# plt.imshow(image, cmap='gray')
# plt.title(f'Label: {labels.item()}')
# plt.show()

img = Image.open('mnist_img_norm.png')
image_array = np.array(img)
img_tens=torch.tensor(image_array)
print(image_array.shape)
