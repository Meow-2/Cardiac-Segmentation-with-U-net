import os
from PIL import Image
import numpy as np

pre_path = 'result_unetpluses_o'
label_path = 'o_labels'
mask_path = './'+pre_path+'/Predict/Mask/'
masks = os.listdir(mask_path)
all_dice = []
for i in range(len(masks)):
    pre = np.array(Image.open(mask_path + masks[i]))
    label = np.array(Image.open('./dataset/'+ label_path + '/' + masks[i]))
    tp = np.count_nonzero((label == 255) & (pre == 1)) 
    fp = np.count_nonzero(pre == 1) - tp
    fn = np.count_nonzero(label == 255) - tp 
    tn = np.count_nonzero((pre == label) & (pre != 1))
    dice = 2*tp/(2*tp+fp+fn)
    print(masks[i] + " Dice:%",dice)
    all_dice.append(dice)
mean_dice = np.mean(all_dice)
with open('./'+pre_path + '/test_dice_'+ str(mean_dice) +'.txt',"w") as f:
    print("Finish")