import torch
import torch.nn as nn
import torch.nn.functional as F

# W/H=[(输入大小-卷积核大小+2*P）/步长]  +1.
# # 反卷积,已知卷积核和卷积结果大小，求卷积输入大小，通过插值来实现
# 输入大小 = (步长*(W/H-1)+卷积核大小-2*P)
# nn.ConvTranspose2d的功能是进行反卷积操作
# nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
# padding(int or tuple, optional) - 输入的每一条边补充0的层数，高宽都增加2*padding
# output_padding(int or tuple, optional) - 输出边补充0的层数，高宽都增加padding
# 输出尺寸计算：
# output = (input-1)stride+outputpadding -2padding+kernelsize

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2,2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self,x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                nn.Conv2d(in_channels,in_channels//2, kernel_size=1,padding=0),
                                nn.ReLU(inplace=True))
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x2 = torch.cat([x2,x1],dim=1)
        return  self.conv(x2)

class unet(torch.nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.feat1 = DoubleConv(3,64)
        self.feat2 = Down(64,128)
        self.feat3 = Down(128,256)
        self.feat4 = Down(256,512)
        self.feat5 = Down(512,1024)
        self.Up1 = Up(1024,512)
        self.Up2 = Up(512,256)
        self.Up3 = Up(256,128)
        self.Up4 = Up(128,64)
        self.classification = nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x):
        feat1 = self.feat1(x)
        feat2 = self.feat2(feat1)
        feat3 = self.feat3(feat2)
        feat4 = self.feat4(feat3)
        feat5 = self.feat5(feat4)
        x = self.Up1(feat5,feat4)
        x = self.Up2(x,feat3)
        x = self.Up3(x,feat2)
        x = self.Up4(x,feat1)
        x = self.classification(x)
        return x

