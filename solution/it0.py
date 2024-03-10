# %%
import torch
from torch.autograd import Variable
import os
import sys
sys.path.append(os.getcwd() + '/../DiffJPEG')
from DiffJPEG import DiffJPEG
# from paq2piq_standalone import MetricModel

# import matplotlib.pyplot as plt
from torchvision import io, utils

# %%
# iterative attack baseline (IFGSM attack)
def attack( image, model=None, metric_range=100, device='cpu',
            eps = 10 / 255,
            iters = 10,
            alpha = 1/255):
    """
    Attack function.
    Args:
    image: (torch.Tensor of shape [1,3,H,W]) clear image to be attacked.
    model: (PyTorch model): Metric model to be attacked. Should be an object of a class that inherits torch.nn.module and has a forward method that supports backpropagation.
    iters: (int) number of iterations. Can be ignored, during testing always set to default value.
    alpha: (float) step size for signed gradient methods. Can be ignored, during testing always set to default value.
    device (str or torch.device()): Device to use in computaions.
    eps: (float) maximum allowed pixel-wise difference between clear and attacked images (in 0-1 scale).
    Returns:
        torch.Tensor of shape [1,3,H,W]: adversarial image with same shape as image argument.
    """
    image = Variable(image.clone().to(device), requires_grad=True)
    additive = torch.zeros_like(image).to(device)
    additive = Variable(additive, requires_grad=True)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.functional.nll_loss
    loss_fn = lambda output, target: 1 - output/metric_range
    # loss_fn = torch.nn.functional.cross_entropy
    target = torch.tensor([1.])
    height = image.size()[2]
    width = image.size()[3]
    jpeg = DiffJPEG(height=height, width=width, differentiable=True, quality=80)
    dmodel = torch.nn.Sequential(jpeg, model)
    # print(model(image))
    for _ in range(iters):
        output = dmodel(image)
        # print("Out:",output)
        # print(additive)
        
        loss = loss_fn(output, target)
        # model.zero_grad()
        # loss.backward()
        # input_grad = image.grad.data
        input_grad = torch.autograd.grad(loss, image)[0]

        # print(torch.sum(input_grad))
        # print(loss)
        # print(output, target, torch.sum(input_grad))
        gradient_sign = input_grad.sign()

        additive = eps*gradient_sign
        # print(additive)
        image = image - additive
    
    res_image = image - additive
    # from random import random
    # utils.save_image(additive, f"{int(random()*10000)}.jpeg")

    res_image = (res_image).data.clamp_(min=0, max=1)
    # print(dmodel(res_image))
    return res_image

# %%
# img = io.read_image('../NIPS_test/lenna.jpeg')

# img = img/255
# plt.imshow(img.T)

# %%
# m = MetricModel('cpu','additonal_files/weights/RoIPoolModel.pth')
# img = img.unsqueeze(0)
# logits = m(img)
