# 用pytorch搭建简单的语义分割(可训练自己的数据集)

#### 灵感来源：[https://blog.csdn.net/weixin_44791964/article/details/102979289](https://blog.csdn.net/weixin_44791964/article/details/102979289)

本博客的搭建的网络源于这位博主采用的keras框架，不过基于本人电脑配置做了一些网络层数的改动。部分引用大佬的代码，其余均为本人原创。

------

### 整体文件目录下排放：

![](博客\3.png)



------

### 1、编码器Mobilenet：

这里也有大佬关于Mobilenet的博客[Mobilenet的介绍](https://blog.csdn.net/weixin_44791964/article/details/102819915)。简单来说Mobilenet利用深度卷积使得数据量大大减少，有助于配置较低的机器，也可以应用到手机上。

```python
import torch
from torch.nn import *
from torch.nn.functional import relu6



#第一个卷积块
class Conv_block(Module):
    def __init__(self,inplanes,outplanes,strides):
        super(Conv_block, self).__init__()
        self.zeropad=ZeroPad2d(padding=1)
        self.conv=Conv2d(inplanes,outplanes,kernel_size=3,stride=strides,padding=0)
        self.BN=BatchNorm2d(outplanes,momentum=0.1)
        # self.relu=ReLU()

    def forward(self,x):
        x=self.zeropad(x)
        x=self.conv(x)
        x=self.BN(x)
        # x=self.relu(x)
        x=relu6(x)

        return x



#除了第一个卷积块的后面的深度卷积块
class depthwise_block(Module):
    def __init__(self,inplanes,outplanes,strides):
        super(depthwise_block, self).__init__()
        self.zeropad=ZeroPad2d(padding=1)
        self.DW=Conv2d(inplanes,inplanes,                      #深度卷积,输入和输出通道一致
                       kernel_size=3,stride=strides,
                       padding=0,groups=inplanes,           #groups=inplanes是实现深度卷积的重点
                       bias=False)
        self.BN_1=BatchNorm2d(inplanes,momentum=0.1)
        self.BN_2=BatchNorm2d(outplanes,momentum=0.1)
        self.conv=Conv2d(inplanes,outplanes,kernel_size=1,stride=1)
        # self.relu=ReLU()

    def forward(self,x):
        x=self.zeropad(x)
        x=self.DW(x)
        x=self.BN_1(x)
        # x=self.relu(x)
        x = relu6(x)
        x=self.conv(x)
        x=self.BN_2(x)
        # x=self.relu(x)
        x=relu6(x)


        return x

    
class Mobilenet(Module):

    cfg_filter=[32,64,128,128,256,256]               #每个block的inplanes、outplanes
    cfg_stride=[1,2,1,2,1]                                      #每个block的strides
    cfg_block=[]                                                           #初始化后的block集成一个列表

    layer_data=[]                                                 #每个block处理后的output

    def __init__(self):
        super(Mobilenet, self).__init__()
        self.conv_block=Conv_block(3,32,2)               #第一个conv block


        self.block_1=depthwise_block(32,64,1)
        self.block_2=depthwise_block(64,128,2)
        self.block_3=depthwise_block(128,128,1)
        self.block_4=depthwise_block(128,256,2)
        self.block_5=depthwise_block(256,256,1)

    def forward(self,inputs):
        x=inputs
        x=self.conv_block(x)

        x=self.block_1(x)
        x=self.block_2(x)
        x=self.block_3(x)
        x=self.block_4(x)
        x=self.block_5(x)

        return x


#测试encoder网络
if __name__ =="__main__":

    model=Mobilenet()
    inputs=torch.randn(1,416,416,3).permute(0,3,1,2)
    # inputs=torch.randn(1,3,416,416)
    # layers_list=model(inputs)
    outputs = model(inputs)
    print("layers_3 shape:" )
    # print(layers_list[2].shape)
    print(outputs.shape)
    
    
```

------

### 2、解码器Segnet：

解码器对应着上面的编码器，目的是**把获得的特征重新映射到比较搭的图片中的每一个像素点，用于每一个像素点的分类**。放大倍数上，和大佬博客不一样的是，本人**把最终的size放大到放入网络的size**，个人认为这样有助于每个像素的特征得到对应。

```python
import torch
import numpy as np
from torch.nn import *
from torch.nn import functional as F
from mobilenet_ import Mobilenet



class Segnet(Module):

    cfg_filter=[256,128,64,32]
    conv_block=[]
    BN_block=[]

    def __init__(self,num_classes):
        super(Segnet, self).__init__()
        self.zeropad=ZeroPad2d(padding=1)
        self.conv_1=Conv2d(256,256,kernel_size=3,padding=0)
        self.conv_2=Conv2d(32,num_classes,kernel_size=3,padding=1)
        self.BN_1=BatchNorm2d(256,momentum=0.1)
        self.upsample=Upsample(scale_factor=2)


        for i in range(len(self.cfg_filter)-1):
            self.conv_block += [Conv2d(self.cfg_filter[i],
                                       self.cfg_filter[i + 1],
                                       kernel_size=3,
                                       padding=0)]

            self.BN_block +=[BatchNorm2d(self.cfg_filter[i+1])]


        self.conv_block=ModuleList(self.conv_block)
        self.BN_block = ModuleList(self.BN_block)

    def forward(self,o):

        #input:52,52,256
        o=self.zeropad(o)
        o=self.conv_1(o)
        o=self.BN_1(o)

        #input:104,104,256
        for j in range(3):
            o=self.upsample(o)
            o=self.zeropad(o)
            o=self.conv_block[j](o)
            o=self.BN_block[j](o)

        outputs=self.conv_2(o)

        return outputs


#编码器和解码器组合
class Airplanesnet(Module):
    def __init__(self,classes1,BATCH_SIZE):
        super(Airplanesnet, self).__init__()
        self.encoder_part=Mobilenet()     #Mobilenet()是从另一个py文件import过来的类
        self.decoder_part=Segnet(classes1)
        self.classes=classes1
        self.batch_size=BATCH_SIZE


    def forward(self,input_1):
        x=self.encoder_part(input_1)
        x=self.decoder_part(x)
        return x

#测试decoder网络
if __name__ =="__main__":


    model=Airplanesnet(classes1=2,BATCH_SIZE=1)
    inputs_1=torch.Tensor(torch.randn(1,3,416,416))
    outputs_1=model(inputs_1)
    # outputs=outputs[3]
    print("outputs shape:" )
    print(outputs_1.shape)
```

**Segnet最后没有进行softmax，因为training用的是CrossEntropyLoss**。

------

### 3、训练自己的数据集training：

本博客采用的是**VOC2012语义分割两类数据集**，并通过自己的代码处理以及数据增强得到看起来为**黑色的**，**每个像素值非0即1(目标像素值为1)**的label。

![](博客\1.png)



![](博客\2.png)

代码如下：

```python
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
            print('[epoch: %d,idex: %2d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/322))  #训练集的数量,可根据数据集调整
            runing_loss=0.0         #收集的loss值清零

        torch.save(model.state_dict(),f='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth') #保存权重


for epoch in range(EPOCH):    #迭代次数
    train(epoch)
```

------

### 4、预测文件predict：

把待测图像放入samples文件夹中，输出结果在outputs文件夹中

![](博客\4.png)

![](博客\5.png)

predict的重点是：**图像从model输出后得到的predict图像经过【第70行pr=predict.argmax(axis=-1)】压缩成一层，每个像素值为种类概率最高的该层的index，再遍历全部像素点与种类index进行匹配，匹配成功则涂上上对应的颜色。**

```python
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
```

预测结果：

![](博客\6.jpg)

------

### 5、语义分割mIoU评测指标：

源码：[https://www.cnblogs.com/Trevo/p/11795503.html](https://www.cnblogs.com/Trevo/p/11795503.html)，即把两个矩阵进行mIoU评测，所以我们之后要做的很简单，就是跟predict相似，img经过model后输出predict，然后与label进行匹配。

```python
from segnet_ import Airplanesnet
import numpy as np
import torch
import argparse
import copy
import cv2


NCLASSES = 2
BATCH_SIZE=1

#文件的加载路径
parser = argparse.ArgumentParser()
parser.add_argument('--val_txt', type=str, default='D:/untitled/.idea/SS_torch/dataset/val.txt', help='about validation')
parser.add_argument('--weights', type=str, default='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth', help='weights')
opt = parser.parse_args()
print(opt)

txt_path = opt.val_txt
weight=opt.weights




__all__ = ['SegmentationMetric']

class SegmentationMetric(object):                 #计算mIoU、accuracy的类
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        acc = round(acc,5)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        mIoU =round(mIoU,4)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))



#读取val.txt中的图片的名称
paths = open("%s" % txt_path, "r")
data = []

for lines in paths:
    path = lines.rstrip('\n')
    data.append(path)



device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")   #检测是否有GPU加速

model=Airplanesnet(NCLASSES,BATCH_SIZE)             #初始化model


model.load_state_dict(torch.load(opt.weights))     #加载权重

model.to(device)


sum_1 = 0  # 累加每张图片val的accuracy
sum_2 = 0  # 累积每张图片Val的mIoU

for i in range(len(data)):

    image = cv2.imread("D:/untitled/.idea/SS_torch/dataset/jpg_right/%s" % data[i] + ".jpg", -1)
    label = cv2.imread("D:/untitled/.idea/SS_torch/dataset/png_right/%s" % data[i] + ".png", -1)


    orininal_h = image.shape[0]               # 读取的图像的高
    orininal_w = image.shape[1]               # 读取的图像的宽

    image = cv2.resize(image, dsize=(416, 416))
    label = cv2.resize(label, dsize=(416, 416))

    label[label >= 0.5] = 1           #label被resize后像素值会改变,调整像素值为原来的两类
    label[label < 0.5] = 0

    image = image / 255.0          # 图像归一化
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)             # 显式的调转维度

    image = torch.unsqueeze(image, dim=0)             # 改变维度,使得符合model input size
    image = image.type(torch.FloatTensor)             # 数据转换,否则报错
    image = image.to(device)              # 放入GPU中计算

    predict = model(image).cpu()


    predict = torch.squeeze(predict)               # [1,1,416,416]---->[1,416,416]
    predict = predict.permute(1, 2, 0)


    predict = predict.detach().numpy()

    prc = predict.argmax(axis=-1)


    #进行mIoU和accuracy的评测
    imgPredict =prc
    imgLabel = label

    metric = SegmentationMetric(2)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    sum_1+=acc
    mIoU = metric.meanIntersectionOverUnion()
    sum_2+=mIoU
    print("%s.jpg :" % data[i])
    print("accuracy:  "+str(acc*100)+" %")
    print("mIoU:  " +str(mIoU))
    print("-------------------")


# 全部图片平均的accuracy和mIoU
sum_1=sum_1/len(data)
sum_2=sum_2/len(data)

sum_1 = round(sum_1,5)
sum_2 = round(sum_2,4)

print("M accuracy:  "+str(sum_1*100)+" %")
print("M mIoU:  " +str(sum_2))
```

评测结果：

![](博客\6.png)

------

### 6、数据增强：

本语义分割代码采用**python的Augmentor库**，但是有个缺点就是**每次增强只能一张图片**，多于一张会**让label和image的形变不对应**，所以代码有点绕，即读取一张图片，**增强后用os把图片移除**，再把处理好的label和image分别放入不同文件夹，方便以上一系列操作。

```python
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
```

------

### 7、关于训练集和验证集的txt文件：

![](博客\7.png)

![](博客\8.png)

```python
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
```

------

### 总结：

**该算法不使用预训练模型**，原因有二：第一，难以找到mobilenet的预训练模型，并且还要进行个性化的修改，较麻烦。第二，模型数据量小，**训练好的权重大小为4MB左右(图片数为357)**，从头开始训练的效果也达到及格线。

![](博客\9.png)