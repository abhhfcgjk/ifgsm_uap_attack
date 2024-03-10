# %%
import torch
from torch.autograd import Variable
from paq2piq_standalone import MetricModel

import matplotlib.pyplot as plt
from torchvision import io

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

    output = model(image)
    print(output)
    
    for i in range(iters):
        ...
        
    res_image = image + additive

    res_image = (res_image).data.clamp_(min=0, max=1)
    return res_image

# %%
# img = io.read_image('public_tests/lenna.jpeg')

# img = img/255
# plt.imshow(img.T)

# %%
# m = MetricModel('cpu','additonal_files/weights/RoIPoolModel.pth')
# img = img.unsqueeze(0)
# logits = m(img)

# %%



