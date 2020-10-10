from segnet_ import Airplanesnet
from PIL import Image
import numpy as np
import torch
import argparse
import cv2
import copy
import os


parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=str, default='D:/untitled/.idea/SS_torch/samples', help='samples')
parser.add_argument('--outputs', type=str, default='D:/untitled/.idea/SS_torch/outputs', help='outputs')
parser.add_argument('--weights', type=str, default='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth', help='weights')
opt = parser.parse_args()
print(opt)


colors = [[0,0,0],[255,0,0]]
NCLASSES = 2
BATCH_SIZE=1

img_way=opt.samples
img_save=opt.outputs


device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")   #检测是否有GPU加速

model=Airplanesnet(NCLASSES,BATCH_SIZE)             #初始化model

model.load_state_dict(torch.load(opt.weights))     #加载权重

model.to(device)     #放入GPU

for jpg in  os.listdir(r"%s" %img_way):

    name = jpg[:-4]
    with torch.no_grad():

        image=cv2.imread("%s" % img_way + "/" + jpg)
        old_image = copy.deepcopy(image)
        old_image = np.array(old_image)
        orininal_h = image.shape[0]       #读取的图像的高
        orininal_w = image.shape[1]       #读取的图像的宽   方便之后还原大小

        image = cv2.resize(image, dsize=(416, 416))         #调整大小
        image = image / 255.0                   #图像归一化
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)                #显式的调转维度


        image = torch.unsqueeze(image, dim=0)           #改变维度,使得符合model input size
        image = image.type(torch.FloatTensor)         #数据转换,否则报错
        image = image.to(device)                      #放入GPU中计算


        predict = model(image).cpu()
        # print(predict.shape)

        predict = torch.squeeze(predict)            #[1,1,416,416]---->[1,416,416]
        predict =predict.permute(1, 2, 0)
        # print(jpg)

        predict = predict.numpy()
        # print(predict.shape)

        pr=predict.argmax(axis=-1)                     #把class数量的层压缩为一层,Z轴上的值概率最高的返回该层index


        seg_img = np.zeros((416, 416,3))        #创造三层0矩阵,方便进行涂色匹配

        #进行染色
        for c in range(NCLASSES):

            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')


        seg_img = cv2.resize(seg_img,(orininal_w,orininal_h))
        seg_img = np.array(seg_img)

        # 原图和效果图叠加
        result = cv2.addWeighted(seg_img, 0.3, old_image, 0.7, 0., old_image, cv2.CV_32F)
        cv2.imwrite("%s/%s" % (img_save, name) + ".jpg", result)
        print("%s.jpg  ------>done!!!" % name)














