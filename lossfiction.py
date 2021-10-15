import torch.nn as nn
import torch.nn.functional as F
# jpg_pre is y_pre,png is y_true,labels is the one-hat of the y_true
def dice_ce_Loss(jpg_pre, png, labels,Smooth = 1e-5):
    """ computational formula：
            dice = (2 * (y_pre ∩ y)) / (y_pre ∪ y)
            miou = dice/(2 - dice)
    """
    n, c, h, w = jpg_pre.size() # png n,h,w   labels n,h,w,c
    # 交叉熵损失
    ce_jpg = F.log_softmax(jpg_pre, dim=1)
    ce_jpg = ce_jpg.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
    ce_png = png.view(-1)
    CE_loss = nn.NLLLoss()(ce_jpg,ce_png)
    #dice_loss
    jpg_pre = F.softmax(jpg_pre,dim = 1).transpose(1,2).transpose(2,3)
    Dice = []
    for i in range(jpg_pre.shape[-1]):
        # 交集
        intersection = jpg_pre[...,i]*labels[...,i]
        intersection  = intersection.sum(axis = [1,2])
        union = jpg_pre[...,i].sum(axis = [1,2]) + labels[...,i].sum(axis = [1,2])
        dice = (2 * intersection + Smooth) / (union + Smooth)
        Dice.append(dice.mean(axis = 0))
    all_loss = CE_loss + 1 - Dice[1]
    return all_loss,Dice[1]

def dice_Loss(jpg_pre, png, labels,Smooth = 1e-5):
    """ computational formula：
            dice = (2 * (y_pre ∩ y)) / (y_pre ∪ y)
            miou = dice/(2 - dice)
    """
    #dice_loss
    jpg_pre = F.softmax(jpg_pre,dim = 1).transpose(1,2).transpose(2,3)
    Dice = []
    for i in range(jpg_pre.shape[-1]):
        # 交集
        intersection = jpg_pre[...,i]*labels[...,i]
        intersection  = intersection.sum(axis = [1,2])
        union = jpg_pre[...,i].sum(axis = [1,2]) + labels[...,i].sum(axis = [1,2])
        dice = (2 * intersection + Smooth) / (union + Smooth)
        Dice.append(dice.mean(axis = 0))
    all_loss = 1 - Dice[1]
    return all_loss,Dice[1]