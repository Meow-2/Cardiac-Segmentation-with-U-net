import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import os 

def rand(a=0, b=1):
    # np.random.rand()返回(0,1)的随机数，故rand返回(a,b)之间的随机数
    return np.random.rand() * (b - a) + a

class Mridataset(Dataset):  #用于加载训练集和验证集
    def __init__(self,path,label_dir,image_size,num_classes,random_data):
        #打开存储图片列表的txt文件
        self.path = path
        self.label_dir = label_dir
        self.image_size = image_size
        self.random_data = random_data
        self.num_classes = num_classes
        #根据划分的训练集验证集测试集来形成样本列表
        self.sample_list = []
        with open(path) as f:
            patient_number = f.readlines()
            for i in range(len(patient_number)):
                patient_number[i] = patient_number[i].split()[0] + '-'
            labels = os.listdir('./dataset/' + label_dir)
            for i in range(len(labels)):
                for j in range(len(patient_number)):
                    if labels[i].startswith(patient_number[j]):
                        self.sample_list.append(labels[i])

    def increase_channels(self,image):
        image = np.array(image)
        image_color = np.zeros(shape=(image.shape[0],image.shape[1],3),dtype=np.float32)
        image_color[...,0] = image[:,:]
        image_color[...,1] = image[:,:]
        image_color[...,2] = image[:,:]
        return image_color
    #数据增强
    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1 - jitter, 1 + jitter)  # 0.7~1.3
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2  # 更改宽高比

        scale = rand(0.25, 2)  # 0.25~2,随机一个缩放规模
        if new_ar < 1:  # new_ar为w/h,将较大的一方scale，保持w/h不变
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)  # 计算缩放后的大小
        image = image.resize((nw, nh), Image.BICUBIC)  # 图像大小变换，三次样条插值
        label = label.resize((nw, nh), Image.NEAREST)  # 图像变换，低质量
        label = label.convert("L")  # 标签转化为灰度图

        # flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)  # 0.5的概率左右翻转

        # place image
        dx = int(rand(0, w - nw))  # 调整粘贴的位置
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))  # paste函数的参数为(粘贴的图片，粘贴的起始点的横坐标，粘贴的起始点的纵坐标）
        image = new_image
        label = new_label

        # distort image                     #对图形进行明暗变化
        hue = rand(-hue, hue)  # -0.1~0.1
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        # 0.5的概率，sat是(1,1.5),0.5的概率，sat是(0.5,1)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)  # 将rgb转化为hsv空间
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        return image_data, label

    def __getitem__(self, item):
        name = self.sample_list[item].split()[0]
        jpg = Image.open('./dataset/images/' + name,'r')
        png = Image.open('./dataset/' + self.label_dir+ '/' + name,'r')
        #是否调用数据增强模块
        if self.random_data and rand(0,1)<0.3:
            jpg,png = self.get_random_data(jpg,png,(int(self.image_size[0]),int(self.image_size[1])))
        else:
            jpg = self.increase_channels(jpg)
        # 将image转化为numpy数组,jpg归一化
        # cv2.imwrite("hsv.jpg",jpg)
        jpg = np.transpose(np.array(jpg), [2,0,1])/255
        png = (np.array(png)/255).astype('int')
        png[png >= self.num_classes-1] = self.num_classes - 1
        # 转化成one_hot的形式
        # c = np.where(png == 0)
        # a = np.where(png == 1)
        # b = np.where(png == 2)
        seg_labels = np.eye(self.num_classes)[png.reshape([-1])]        #利用eye生成对角矩阵
        seg_labels = seg_labels.reshape((int(self.image_size[0]), int(self.image_size[1]), self.num_classes))

        jpg = torch.from_numpy(jpg).type(torch.FloatTensor)
        png = torch.from_numpy(png).type(torch.FloatTensor).long()
        seg_labels = torch.from_numpy(seg_labels).type(torch.FloatTensor)

        # print("a",seg_labels[a[0][0],a[1][0],:])
        # print("b",seg_labels[b[0][0],b[1][0],:])
        # print("c",seg_labels[c[0][0],c[1][0],:])

        # ToTensor,改变通道位置
        # seg_labels = trans.ToTensor()(seg_labels)
        # jpg = trans.ToTensor()(jpg.reshape(256,256,1))
        # png = trans.ToTensor()(png.reshape(256,256,1))
        return jpg,png,seg_labels

    def __len__(self):
        return len(self.sample_list)

# train_set = Mridataset("../dataset/train.txt",(256,256),3,True)
# val_set = Mridataset("../dataset/val.txt",(256,256),3,False)
# jpg1,png1,ont_hat1 = train_set[0]
# jpg,png,one_hat = val_set[0]
# png1[png1==1]=255
# png1[png1==2]=128
# png[png==1]=255
# png[png==2]=128
# Image.fromarray(jpg1*255).show()
# Image.fromarray(png1).show()
# Image.fromarray(jpg*255).show()
# Image.fromarray(png).show()
# np.where(png == 2)
