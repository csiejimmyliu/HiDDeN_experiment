from PerceptualSimilarity.src.loss.loss_provider import LossProvider
provider = LossProvider()
loss_function = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')

import torch
img0 = torch.zeros(1,3,64,64)
img1 = torch.ones(1,3,64,64)
loss = loss_function(img0, img1)
print(loss)