import os
import numpy as np
import cv2
from PIL import Image

images_path = "./images"
o_contours_path = "./o_labels"
i_contours_path = "./i_labels"
o_color = [0,255,0]
i_color = [255,0,0]

o_contours = os.listdir(o_contours_path)
for i in range(len(o_contours)):
    #读取image和内外膜标注
    image = cv2.imread(images_path+'/'+o_contours[i],0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    o_mask = cv2.imread(o_contours_path+'/'+o_contours[i],0)
    i_mask = cv2.imread(i_contours_path+'/'+o_contours[i],0)
    image_copy = image.copy()
    #标注图像
    image_copy[...,:][o_mask == 255] = o_color
    image_copy[...,:][i_mask == 255] = i_color

    image_copy = Image.fromarray(np.uint8(image_copy))
    image = Image.fromarray(np.uint8(image))
    #图像结合
    image = Image.blend(image, image_copy, 0.3)
    image.save("./show/"+o_contours[i])
