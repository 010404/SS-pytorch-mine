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
        # x=x[2]
        x=self.decoder_part(x)
        # x=x.view(self.batch_size,2,43264)
        # x=F.softmax(x,dim=1)

        return x

#测试decoder网络
if __name__ =="__main__":


    model=Airplanesnet(classes1=2,BATCH_SIZE=1)
    inputs_1=torch.Tensor(torch.randn(1,3,416,416))
    outputs_1=model(inputs_1)
    # outputs=outputs[3]
    print("outputs shape:" )
    print(outputs_1.shape)
































































































