import os
import random

val_percent = 0.1
train_percent = 0.9
imagepath = 'dataset/jpg_right'
txtsavepath = 'dataset'
total_img = os.listdir(imagepath)

num = len(total_img)
list=range(num)

tv = int(num * val_percent)     #验证个数
tr = int(num-tv)             #训练个数


num_trainval = random.sample(list, tv)                #随机获取tv个片段

num_train = random.sample(list, tr)                #随机获取tr个片段


ftrain = open('dataset/train.txt', 'w')
fval = open('dataset/val.txt', 'w')

for i in range(num):
    name = total_img[i][:-4] + '\n'      #提取名字+转行
    if i in num_train:
        ftrain.write(name)

    else:
        fval.write(name)

    print("True")

print(i+1)

ftrain.close()
fval.close()





















