import torch
import time
from torch import optim
from utils.dataloader import Mridataset
from torch.utils.data import DataLoader
from models.vgg_unet import vgg_unet
from tqdm import tqdm
import numpy as np
from lossfiction import dice_ce_Loss,dice_Loss
from utils.save_load import save_as_file,load_from_file,delete_file

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(epoch,Epoch,dataloader):
    running_loss = 0
    dataloader_size = len(dataloader)
    # first_batch = next(iter(dataloader))
    # dataloader_size = 50
    with tqdm(total=dataloader_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        # for i,data in enumerate([first_batch]*50):
        for i,data in enumerate(dataloader):
            jpgs,pngs,labels = data
            if cuda:
                jpgs = jpgs.cuda()
                pngs = pngs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            #模型预测
            jpgs = model(jpgs)
            #计算loss
            loss,dice = dice_ce_Loss(jpgs,pngs,labels)
            #反向传播
            loss.backward()
            #梯度下降
            optimizer.step()
            #计算这一个minibatch的loss并计入到这个epoch的loss之中
            running_loss+= loss.item()
            pbar.set_postfix(**{'total_loss': running_loss / (i + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    return running_loss/dataloader_size

def test(train_loss,epoch,Epoch,dataloader):
    val_loss = 0
    val_dice = 0
    dataloader_size = len(dataloader)
    print("Validation Epoch:",epoch+1)
    with tqdm(total=dataloader_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for i, data in enumerate(dataloader):
            jpgs, pngs, labels = data
            with torch.no_grad():
                if cuda:
                    jpgs = jpgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                jpgs = model(jpgs)
                loss,dice = dice_ce_Loss(jpgs,pngs,labels)
                val_loss += loss.item()
                val_dice += dice.item()
            pbar.set_postfix(**{'val_loss': val_loss/ (i + 1),
                                'val_dice': val_dice / (i + 1),
                                    'lr': get_lr(optimizer)})
            pbar.update(1)
    print("Finish Validation Epoch:",epoch+1)
    val_loss = val_loss / dataloader_size
    val_dice = val_dice / dataloader_size
    print('Total Loss: %.4f || Val Loss: %.4f ' % (train_loss,val_loss))
    print('Saving state, epoch:', str(epoch + 1))
    torch.save(model.state_dict(), './'+save_path+'/logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f-Val_Dice%.4f.pth' % ((epoch + 1), train_loss,val_loss,val_dice))
    return val_loss,val_dice

#--------------------参数----------------------#
#linux下相对路径报错时,补全绝对路径
path = ""
batch_size = 2
lr = 1e-4
#冻结训练代数与解冻训练代数,共100代
Freeze_Epoch = 50
Unfreeze_Epoch = 150
cuda =True
pin_memory =True
num_workers = 8

num_classes = 0
contours_type = "o_labels"
model = vgg_unet(True,num_classes) #模型
save_path = "result_vggunet_o"
dice_ce_Loss = dice_Loss
optimizer = optim.Adam(model.parameters(),lr) #优化器
#--------------------------------------------#


#加载数据集
# Mridataset(数据列表路径,图片大小,分类种类,是否数据增强)
train_set = Mridataset("dataset/train.txt", contours_type, (256, 256), num_classes, False)
val_set = Mridataset("dataset/val.txt", contours_type,(256, 256), num_classes, False)

train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
                                drop_last=True)
val_loader = DataLoader(dataset=val_set,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory,
                                drop_last=True)

if __name__ == '__main__':
    #使用迭代器去first_batch,看是否能过拟合
    # first_batch = next(iter(train_loader))
    trainloss_list = "./"+save_path+"/loss_graph/trainloss_list.txt"
    valoss_list = "./"+save_path+"/loss_graph/valoss_list.txt"
    valdice_list = "./"+save_path+"/loss_graph/valdice_list.txt"
    lr_list = "./"+save_path+"/loss_graph/lr_list.txt"
    delete_file([trainloss_list,valoss_list,valdice_list,lr_list])
    np.save("./"+save_path+"/loss_graph/Freeze_Epoch.npy",np.arange(0,Freeze_Epoch))
    np.save("./"+save_path+"/loss_graph/Unfreeze_Epoch.npy",np.arange(Freeze_Epoch,Unfreeze_Epoch+Freeze_Epoch))
    if cuda:
        model = model.cuda()
    start = time.time()
    for param in model.vgg.parameters():        #冻结backbone
        param.requires_grad = False
    for epoch in range(Freeze_Epoch):# 0~49
        save_as_file(get_lr(optimizer),lr_list)
        train_loss = train(epoch,Freeze_Epoch,train_loader)
        save_as_file(train_loss,trainloss_list)
        val_loss,val_dice  = test(train_loss,epoch,Freeze_Epoch,val_loader)
        save_as_file(val_loss, valoss_list)
        save_as_file(val_dice, valdice_list)
    for param in model.vgg.parameters():        #解冻backbone
        param.requires_grad = True
    for epoch in range(Freeze_Epoch,Freeze_Epoch+Unfreeze_Epoch): #50~100
        save_as_file(get_lr(optimizer),lr_list)
        train_loss =  train(epoch,Freeze_Epoch+Unfreeze_Epoch,train_loader)
        save_as_file(train_loss,trainloss_list)
        val_loss,val_dice = test(train_loss,epoch,Freeze_Epoch+Unfreeze_Epoch,val_loader)
        save_as_file(val_loss,valoss_list)
        save_as_file(val_dice,valdice_list)
    end = time.time()
    trainning_time = end - start
    f = open("./"+save_path+"/vgg_unet_trainning_time"+str(trainning_time)+".txt",'w')
    f.close()