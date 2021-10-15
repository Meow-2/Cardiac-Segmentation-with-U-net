import torch
import torch.nn as nn
import torchvision.models as models
class conv2dxN(nn.Module):
    def __init__(self,in_channels,out_channels,n):
        super(conv2dxN, self).__init__()
        if n == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        if n == 3 :
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    def forward(self,x):
        return self.conv(x)

class vgg_backbone(nn.Module):
    def __init__(self):
        super(vgg_backbone, self).__init__()
        self.feat1 = conv2dxN(3, 64, 2)
        self.feat2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv2dxN(64, 128, 2)
        )
        self.feat3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv2dxN(128, 256, 3)
        )
        self.feat4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv2dxN(256, 512, 3)
        )
        self.feat5 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv2dxN(512, 512, 3)
        )

    def forward(self,x):            #(3,256,256)
        feat1 = self.feat1(x)       #(64,256,256)
        feat2 = self.feat2(feat1)   #(128,128,128)
        feat3 = self.feat3(feat2)   #(256,64,64)
        feat4 = self.feat4(feat3)   #(512,32,32)
        feat5 = self.feat5(feat4)   #(512,16,16)
        return feat1,feat2,feat3,feat4,feat5

    def load_from_pretrain(self):
        model_dict = self.state_dict()  # 模型字典
        pretrained_dict = torch.load("./models/vgg16_bn-6c64b313.pth")  # 预训练字典

        model_key = list(model_dict.keys())
        pretrained_key = list(pretrained_dict.keys())
        pretrained_weight = list(pretrained_dict.values())
        t = 0
        for i in range(len(model_key)):
            if 'num_batches_tracked' in model_key[i]:
                i = i + 1
            else:
                model_dict[model_key[i]] = pretrained_weight[t]
                i = i + 1
                t = t + 1
        self.load_state_dict(model_dict, strict=False)

class Up(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels,in_channels//2, kernel_size=1,padding=0),
                                nn.ReLU(inplace=True))
        self.conv = conv2dxN(in_channels, out_channels,2)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = torch.cat([x2,x1],dim=1)
        return  self.conv(x2)

class vgg_unet(nn.Module):
    def __init__(self,pretrained,num_classes):
        super(vgg_unet, self).__init__()
        self.vgg = vgg_backbone()
        if pretrained:
            self.vgg.load_from_pretrain()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv2dxN(1024,512,2)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)
        self.conv2 = nn.Conv2d(64,num_classes,kernel_size=1)
    def forward(self,x):
        feat1,feat2,feat3,feat4,feat5 = self.vgg(x)
        feat5 = self.upsample1(feat5)   #n,512,32,32
        x = torch.cat([feat4,feat5],dim=1)  #n,1024,32,32
        x = self.conv1(x)               #n,512,32,32
        x = self.up2(x, feat3)           #n,256,64,64
        x = self.up3(x, feat2)          #n,128,128,128
        x = self.up4(x, feat1)          #n,64,256,256
        x = self.conv2(x)               #n,num_classes,256,256
        return x