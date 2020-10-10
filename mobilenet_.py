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












































