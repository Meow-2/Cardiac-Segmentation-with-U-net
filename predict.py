from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.Unet import unet
from models.vgg_unet import vgg_unet
from models.Unetpluses import unetpluses
from utils.dataloader import Mridataset
import time
import os

path = "result_vggunet_o"
contours_type = "o_labels"
num_classes = 2
model_path = "./"+path+"/logs/Epoch168-Total_Loss0.0092-Val_Loss0.0758-Val_Dice0.9242.pth"
model = vgg_unet(True,num_classes)
cuda = True
num_workers = 0
pin_memory = True

model.load_state_dict(torch.load(model_path))
val_set = Mridataset("dataset/test.txt",contours_type,(256, 256),num_classes, False)
val_loader = DataLoader(dataset=val_set,batch_size=1,shuffle=False,num_workers=num_workers,pin_memory=pin_memory,
                                drop_last=False)
if __name__ == '__main__':
    if contours_type == "i_labels":
        color = [255,0,0]
    else:
        color = [0,255,0]
    predict_time_list = []
    picture_name = []
    with open("./dataset/test.txt") as f:
        patient_number = f.readlines()
        for i in range(len(patient_number)):
            patient_number[i] = patient_number[i].split()[0] + '-'
        labels = os.listdir('./dataset/' + contours_type)
        for i in range(len(labels)):
            for j in range(len(patient_number)):
                if labels[i].startswith(patient_number[j]):
                    picture_name.append(labels[i])
    for i,data in enumerate(val_loader):
        jpg,png,labels = data
        jpg_copy = jpg
        n,c,h,w = jpg.shape #n,c,h,w,原始图片通道数
        predict_start = time.time()
        with torch.no_grad():
            if cuda:
                starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                jpg = jpg.cuda()
                model = model.cuda()
                starter.record()
            jpg = model(jpg)
            if cuda:
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            predict_end = time.time()
            jpg = F.softmax(jpg,dim = 1).transpose(1,2).transpose(2,3)
        if cuda:
            jpg = jpg.cpu()
        jpg = jpg.numpy()   #n,h,w,c ->h,w,c,预测后通道数变为了类别
        jpg = jpg.reshape(h,w,num_classes)
        png = png.numpy()   #n,h,w  ->h,w
        png = png.reshape(h,w)
        jpg = np.argmax(jpg,axis = 2)
        jpg_copy = jpg_copy.transpose(1,2).transpose(2,3).numpy().reshape(h,w,c)
        jpg_copy = jpg_copy*255

        seg_img = jpg_copy.copy()
        seg_img[...,:][jpg == 1] = color

        mask = Image.fromarray(np.uint8(jpg))
        seg_img = Image.fromarray(np.uint8(seg_img))
        jpg_copy = Image.fromarray(np.uint8(jpg_copy))
        image = Image.blend(jpg_copy, seg_img, 0.3)

        image.save("./"+path+"/Predict/"+picture_name[i])
        # mask.save("./"+path+"/Predict/Mask/"+picture_name[i])

        predict_time = predict_start - predict_end
        if cuda:
            predict_time = curr_time
        predict_time_list.append(predict_time)
        print("Has save "+picture_name[i]+" Spend time(if cpu s,gpu ms): Predict-" + str(predict_time))
    predict_time_list = np.array(predict_time_list)
    np.save("./"+path+"/Predict/predict_time.npy",predict_time_list)
    print("Finished!")