import torch
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from torch.nn import *
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader

from segnet_ import Airplanesnet



BATCH_SIZE1=1              #训练的batch_size
BATCH_SIZE2=1               #验证的batch_size
NUM_CLASSES=2                 #分割的种类数
LR=1e-4                            #学习率
EPOCH=20                     #迭代次数


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',action='store_true',default=True,help='whether use gpu')
parser.add_argument('--train_txt', type=str, default='D:/untitled/.idea/SS_torch/dataset/train.txt', help='about trian')
parser.add_argument('--val_txt', type=str, default='D:/untitled/.idea/SS_torch/dataset/val.txt', help='about validation')

opt = parser.parse_args()
print(opt)

txt_1 = opt.train_txt
txt_2 = opt.val_txt



#自定义数据集的类
class AirplanesDataset(Dataset):

    def __init__(self,txt_path):
        super(AirplanesDataset, self).__init__()


        paths=open("%s" % txt_path,"r")
        data=[]

        for lines in paths:
            path=lines.rstrip('\n')
            data.append(path)

        self.data=data
        self.len=len(data)


    def __getitem__(self, index):


        image=cv2.imread("D:/untitled/.idea/SS_torch/dataset/jpg_right/%s" %self.data[index]+".jpg",-1)
        label = cv2.imread("D:/untitled/.idea/SS_torch/dataset/png_right/%s"%self.data[index] +".png" , -1)


        image = cv2.resize(image, dsize=(416, 416))
        label = cv2.resize(label, dsize=(416, 416))


        image=torch.from_numpy(image)
        label=torch.from_numpy(label)

        image = image / 255.0            #归一化

        label[label>=0.5]=1                 #label被resize后像素值会改变,调整像素值为原来的两类
        label[label < 0.5] = 0


        image=image.permute(2,0,1)        #调整图像维度,方便载入model


        return image,label


    def __len__(self):

        return self.len


train_dataset = AirplanesDataset(txt_1)  # 训练集




# 加载训练数据集,并且分好mini-batch
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE1,
                          shuffle=True)


criterion = CrossEntropyLoss()  # Loss



model=Airplanesnet(NUM_CLASSES,BATCH_SIZE1)



optimizer = Adam(model.parameters(),  # 优化器
                lr=LR)


device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")   #检测是否有GPU加速
model.to(device)       #网络放入GPU里加速


model.load_state_dict(torch.load('D:/untitled/.idea/SS_torch/weights/SS_weight_2.pth'))


#train函数
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):         #0是表示从0开始
        image,label=data

        # label = torch.squeeze(label)
        # lll=label.numpy()
        # print(lll.shape)

        # f = open('D:/untitled/.idea/SS_torch/dataset/111.txt', 'w')
        #
        # for x in range(lll.shape[0]):
        #     f.write('\n')
        #     for y in range(lll.shape[1]):
        #
        #         f.write(str(lll[x, y,]))


        # # label=label.view(BATCH_SIZE1,416,416)
        # label = torch.unsqueeze(label, dim=0)


        image,label=image.to(device),label.to(device)            #数据放进GPU里
        optimizer.zero_grad()                  #优化器参数清零

        #forword+backward+update
        image=image.type(torch.FloatTensor)        #转化数据类型,不转则会报错
        image=image.to(device)
        outputs=model(image)
        loss=criterion(outputs,label.long())        #进行loss计算

        lll=label.long().cpu().numpy()             #把label从GPU放进CPU


        loss.backward(retain_graph=True)                  #反向传播(求导)
        optimizer.step()            #优化器更新model权重

        running_loss+=loss.item()       #收集loss的值


        if batch_idx % 100 ==99:
            print('[epoch: %d,idex: %2d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/322))
            runing_loss=0.0         #收集的loss值清零

        torch.save(model.state_dict(),f='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth') #保存权重



for epoch in range(EPOCH):    #迭代次数
    train(epoch)
































