import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
from tqdm import tqdm
import random
from PIL import Image
import cv2


# Utils for baseline training
def center_crop(image):
  center = image.shape[0] / 2, image.shape[1] / 2
  if center[1] < 256 or center[0] < 256:
    return cv2.resize(image, (256, 256))
  x = center[1] - 128
  y = center[0] - 128

  return image[int(y):int(y+256), int(x):int(x+256)]


class MyCustomDataset(Dataset):
    def __init__(self, 
                 path_gt,
                 device='cpu'
                ):
        
        self._items = [] 
        self._index = 0
        self.device = device
        dir_img = sorted(os.listdir(path_gt))
        img_pathes = dir_img

        for img_path in img_pathes:
          self._items.append((
            os.path.join(path_gt, img_path)
          ))
        random.shuffle(self._items)

    def __len__(self):
      return len(self._items)

    def __getitem__(self, index):
      gt_path = self._items[index]
      image = Image.open(gt_path).convert('RGB')
      image = np.array(image).astype(np.float32) 

      image = center_crop(image)

      image = image / 255.
      image = transforms.ToTensor()(image)
      y = image.to(self.device)
      return y


# Baseline UAP training fuction
def train(metric_model, path_train, batch_size=8, metric_range=100, device='cpu'):
    """
    UAP adversarial patch training function.
    Args:
    model: (PyTorch model) Metric model to be attacked. Should be an object of a class that inherits torch.nn.module and has a forward method that supports backpropagation.
    path_train: (str) Path to train dataset (Directory with images).
    batch_size: (int) Batch size to train UAP with.
    device: (str or torch.device()) Device to use in computations.
    metric_range: (float) Approximate metric value's range.
    Returns:
        np.ndarray of shape [H,W,3]: UAP patch
    """
    ds_train = MyCustomDataset(path_gt=path_train, device=device)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    eps = 0.1
    lr = 0.001
    n_epoch = 5
    # You can also try random noise
    universal_noise = torch.zeros((1, 3, 256, 256)).to(device)
    universal_noise += 0.0001
    universal_noise = Variable(universal_noise, requires_grad=True)
    optimizer = torch.optim.Adam([universal_noise], lr=lr)
    for epoch in range(n_epoch):
        total_loss = 0
        # Iterate over dl_train, optimize patch, update total_loss (sum/mean of epoch losses)
        # <YOUR CODE HERE>
        ...
        print(f'[{epoch} epoch] Total loss: {total_loss}')
    return universal_noise.squeeze().data.cpu().numpy().transpose(1, 2, 0)


# User-defined function to read pretrained patch
def read_uap_patch(trained_data_path='../uap_trained_data/pretrained_uap_paq2piq.png'):
    """
    Function to read pretrained patch.
    Args:
    trained_data_path: (str) path to your pretrained UAP.
    Returns:
        np.ndarray of shape [H,W,3]: additive that will be passed to attack() during testing.
    """
    uap = cv2.imread(trained_data_path)
    uap = cv2.cvtColor(uap, cv2.COLOR_BGR2RGB)
    uap = uap.astype('float32') / 255.
    uap -= 0.5
    return uap


# Baseline UAP attack function. Should apply pretrained adversarial additive to image
def attack(image, uap_patch, device='cpu',
            eps = 10 / 255,
            ):
    """
    Attack function.
    Args:
    image: (torch.Tensor of shape [1,3,H,W]) clear image to be attacked.
    uap_patch: adversarial additive read with read_uap_patch(). Should be same for all images.
    device (str or torch.device()): Device to use in computaions.
    eps: (float) maximum allowed pixel-wise difference between clear and attacked images (in 0-1 scale).
    Returns:
        torch.Tensor of shape [1,3,H,W]: adversarial image with same shape as image argument.
    """
    image = image.to(device)

    h, w = image.shape[2], image.shape[3]
    uap_h, uap_w = uap_patch.shape[0], uap_patch.shape[1]

    # Resize UAP patch or tile it to match image resolution, then move it to device
    uap_resized = ...

    # UAP pixels after baseline training are in [-0.1, 0.1] range, so (10 * eps) multiplier will limit them to [-eps, eps]
    uap_multiplier = 10 * eps
    attacked_image = image + uap_resized * uap_multiplier
    return torch.clamp(attacked_image, 0, 1)