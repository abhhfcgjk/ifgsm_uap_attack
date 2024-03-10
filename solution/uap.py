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

import sys
# sys.path.append(os.getcwd() + '/../DiffJPEG')
# from DiffJPEG import DiffJPEG

###### UTILS
# Standard libraries
import numpy as np
# PyTorch
import torch
import torch.nn as nn

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
###### UTILS


# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
# import utils


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
    #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result



class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)
    

class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor =  nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float() )
        
    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image*255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']
    

# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
# import utils


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """
    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """
    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self):
        super(block_merging, self).__init__()
        
    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """
    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """
    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        #result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """
    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()
        
        self.height, self.width = height, width
        
    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height/2), int(self.width/2)                
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width                
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255*torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image/255
    

# Pytorch
import torch
import torch.nn as nn
# Local
# from modules import compress_jpeg, decompress_jpeg
# from utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered



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
  # batch_size = 1
  ds_train = MyCustomDataset(path_gt=path_train, device=device)
  dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
  eps = 0.1
  lr = 0.001
  alpha = 5 # 50
  n_epoch = 5
  # target = torch.tensor([1.])
  # You can also try random noise
  # loss_fn = torch.nn.CrossEntropyLoss()
  loss_fn = lambda output, target: 1 - output/metric_range
  # loss_fn = lambda output,target: (1)*torch.sum(output*torch.log(target))
  height = 256
  width = 256
  universal_noise = torch.zeros((1, 3, height, width)).to(device)
  universal_noise += 0.0001
  universal_noise = Variable(universal_noise, requires_grad=True)

  optimizer = torch.optim.Adam([universal_noise], lr=lr)


  jpeg = DiffJPEG(height=height, width=width, differentiable=True, quality=80)
  model = torch.nn.Sequential(jpeg, metric_model)

  for epoch in range(n_epoch):
    total_loss = 0
    i=0
      # Iterate over dl_train, optimize patch, update total_loss (sum/mean of epoch losses)
    for batch in dl_train:
      print(batch.size(), type(batch))
      target = Variable(torch.tensor([100.]), requires_grad=True)
      image = Variable(batch + universal_noise, requires_grad=True)

      output = model(image)
      
      optimizer.zero_grad()
      loss_batch = loss_fn(output, target)
      # print("LOSS: ",loss_batch)
      # print("OUT: ", output)
      # for i in range(batch_size):
      #   print("L:",loss_batch[i])
      #   loss_batch[i].backward()
      # loss_batch.backward()
      loss_batch = torch.mean(loss_batch)
      loss_batch.backward()
      # print(loss_batch)
      

      # grads = torch.autograd.grad(loss_batch, image)
      # print(grads)
      # universal_noise.data = universal_noise.data - p_image[i].grad.data
      #   print(torch.sum(universal_noise))
      optimizer.step()
      # print("Grad:",image.grad.data)
      # print(image.grad.size(), universal_noise.size())
      for grad in image.grad.data:
        universal_noise.data = universal_noise.data - alpha*grad
      universal_noise.data = torch.clamp(universal_noise.data, -eps, eps)

      total_loss += torch.sum(loss_batch)
      i += batch_size
      # if i==10:
      #   break
      # i += 1
      # print(i)
    total_loss /= i
    print(f'[{epoch} epoch] Total loss: {total_loss}')
  # print(universal_noise.data.numpy().shape, universal_noise.size())
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
  # print("PATH: ", uap_patch)
  # uap_img = Image.open(uap_patch)

  h, w = image.shape[2], image.shape[3]
  uap_h, uap_w = uap_patch.shape[0], uap_patch.shape[1]

  # rescale = transforms.Resize((h, w))
  rescaled = cv2.resize(uap_patch, (h,w))

  # Resize UAP patch or tile it to match image resolution, then move it to device
  # uap_resized = torch.from_numpy(rescaled)
  uap_resized = transforms.ToTensor()(rescaled)

  # UAP pixels after baseline training are in [-0.1, 0.1] range, so (10 * eps) multiplier will limit them to [-eps, eps]
  uap_multiplier = 10 * eps
  # print(image.size(), uap_resized.size(), rescaled.shape)
  attacked_image = image + uap_resized * uap_multiplier
  
  return torch.clamp(attacked_image, 0, 1)