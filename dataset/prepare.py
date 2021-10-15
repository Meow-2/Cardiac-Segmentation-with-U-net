import os
import numpy as np
import cv2
from PIL import Image

images_path = "./images"
contours_path = "D:/Sunnybrook_Cardiac_Data/data/scd_manualcontours/SCD_ManualContours"
coutour_type = 'o'

images = os.listdir(images_path)
for i in range(len(images)):

        #读取image生成mask
        image = cv2.imread(images_path+'/'+images[i],0)
        mask = np.zeros_like(image, dtype='uint8')
        #获取取image的contour地址并读取
        no_file_extensions  = images[i].split('.')[0]
        contour_dir_txt = no_file_extensions.split('-IM')
        contour_dir = contour_dir_txt[0] + '/contours-manual/IRCCI-expert'
        contour_txt = 'IM' + contour_dir_txt[1] + '-'+coutour_type+'contour-manual.txt'
        contour = os.path.join(contours_path,contour_dir,contour_txt).replace('\\', '/')
        try:
            contour_coords = np.loadtxt(contour, delimiter=' ').astype('int')
        except IOError:
            with open('./no_'+coutour_type+'contour.txt','a') as f:
                f.write('/'+ contour_dir + '/' + contour_txt + '\n')
        else:
            #对mask进行区域填充
            cv2.fillPoly(mask, [contour_coords], 255)
            mask = Image.fromarray(mask)
            mask.save('./o_labels/'+ images[i])


