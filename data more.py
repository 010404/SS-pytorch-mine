import os
import cv2
import argparse
import Augmentor



#文件路径
parser = argparse.ArgumentParser()
parser.add_argument('--Images', type=str, default='D:/untitled/.idea/SS_torch/Augmentor_img', help='true picture')
parser.add_argument('--final', type=str, default='D:/untitled/.idea/SS_torch/Augmentor_img/output', help='final picture')
parser.add_argument('--Masks', type=str, default='D:/untitled/.idea/SS_torch/Augmentor_mask', help='Mask picture')
parser.add_argument('--jpg_right', type=str, default='D:/untitled/.idea/SS_torch/dataset/jpg_right', help='final picture')
parser.add_argument('--png_right', type=str, default='D:/untitled/.idea/SS_torch/dataset/png_right', help='final masks')
parser.add_argument('--transtxt', type=str, default='D:/untitled/.idea/SS_torch/dataset/trans.txt', help='transtxt')
opt = parser.parse_args()
print(opt)

txt=opt.transtxt

paths = open("%s" % txt, "r")
data = []

for lines in paths:
    path = lines.rstrip('\n')
    data.append(path)


imgway_1=opt.Images
imgway_2=opt.final


JPG_RIGHT=opt.jpg_right
PNG_RIGHT=opt.png_right


#for循环命名需要
n1 = 1
n2 = 1



#进行数据增强
for index in range(len(data)):

    #读取需要增强的image和label
    image = cv2.imread("D:/untitled/.idea/SS_torch/dataset/jpg/%s" % data[index] + ".jpg", -1)
    mask = cv2.imread("D:/untitled/.idea/SS_torch/dataset/png/%s" % data[index] + ".png", -1)

    #保存至数据增强指定的文件夹中
    cv2.imwrite("%s/%s.jpg" % (imgway_1, data[index]) ,image)
    cv2.imwrite("%s/%s.jpg" % (opt.Masks, data[index]) , mask)


    #数据增强主体
    p = Augmentor.Pipeline(opt.Images)     #读取image
    p.ground_truth(opt.Masks)     #读取label,使得label和对应的image进行相同变化的augmentor
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)  #旋转图片,左边最大旋转度,右边最大旋转度
    p.shear(probability=1,max_shear_left=15,max_shear_right=15)       #随机区域形变
    p.flip_left_right(probability=0.5)        #按概率左右翻转
    p.zoom_random(probability=0.5, percentage_area=0.8)       #按概率放大图片
    p.flip_top_bottom(probability=0.5)        #按概率上下翻转
    p.sample(3) #产生3张图片



    os.remove("%s/%s.jpg"%(imgway_1,data[index]))          #去除原来的img,防止mask和img不匹配
    os.remove("%s/%s.jpg" % (opt.Masks, data[index]))      #去除原来的mask,防止mask和img不匹配



#将数据增强后的img和mask进行对应改名并移动到制定的文件夹中
for filename in os.listdir(r"%s" % imgway_2):

    name = filename[:9]
    if name =="Augmentor":         #该图片是image

        name_1 = []  # 把image的数字名称放入列表
        name_1.append(filename[23:34])      #截取数字+格式

        img = cv2.imread("%s" % imgway_2 + "/" + filename,-1)
        name1_1 = name_1[0]
        name2_1 = name1_1[:-6]+str(n1)+ name1_1[6:]           #图片在原来名称基础上改名
        cv2.imwrite("%s/%s" % (JPG_RIGHT, name2_1 )+".jpg", img)
        n1+=1

        if n1==4:           #防止改名出现错误
            n1=1


    else:                           #该图片是mask
        name_2 = []  # 把mask的数字名称放入列表
        name_2.append(filename[31:42])   #截取数字+格式


        img_2 = cv2.imread("%s" % imgway_2 + "/" + filename, -1)
        name1_2 = name_2[0]
        name2_2 = name1_2[:-6] + str(n2) + name1_2[6:]          #图片在原来名称基础上改名
        cv2.imwrite("%s/%s" % (PNG_RIGHT, name2_2)+".png", img_2)
        n2 += 1

        if n2==4:         #防止改名出现错误
            n2=1






























