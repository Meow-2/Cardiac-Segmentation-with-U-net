import torchsummary as summary
from models.Unet import unet
from models.Unetpluses import unetpluses
from models.vgg_unet import vgg_backbone,vgg_unet
import torch

Gpu_on = False #torch.cuda.is_available()
device = torch.device("cuda" if Gpu_on else "cpu")
model = unet(3).to(device)
summary.summary(model,(3,256,256),device="cuda" if Gpu_on else "cpu")
#
# import torch
# import torch.nn as nn
# from models.vggunet import vgg16_bn_backbone
# import torchvision.models as models
# mod = models.vgg16_bn(pretrained=False)
# model = vgg16_bn_backbone()
# # print(model)
# # print(mod.features)
# for name, module in mod._modules.items():
#         print (name," : ",module)
# print('##########')
# for name, module in model._modules.items():
#         print (name," : ",module)