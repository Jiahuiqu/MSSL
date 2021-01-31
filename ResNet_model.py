import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
    )

    return blk


def deconv_block(in_channels, out_channels,stride=2):
    blk = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride,padding=1,output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
    )
    return blk


class ResBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            conv_block(in_channel,in_channel),
            conv_block(in_channel,out_channel)
        )

    def forward(self,input):
        out = self.net(input)
        out = torch.add(out,input)
        return out


class ResNet(nn.Module):

    def __init__(self,in_channel,out_channel):

        super(ResNet, self).__init__()

        self.conv1 = conv_block(in_channel,256)
        self.res1 = nn.Sequential(
            ResBlock(256,256),
            ResBlock(256,256)
        )
        self.deconv1 = deconv_block(256,128)

        self.conv2 = conv_block(128+in_channel,256)
        self.res2  = nn.Sequential(
            ResBlock(256,256),
            ResBlock(256,256)
        )
        self.deconv2 = deconv_block(256,128)

        self.conv3 = conv_block(128 + in_channel,256)
        self.res3 = nn.Sequential(
            ResBlock(256,256),
            ResBlock(256,256)
        )
        self.transition = nn.Conv2d(256,out_channel,kernel_size=3,padding=1,stride=1)

    def forward(self,input1,input2,input3):

        out = self.conv1(input1)
        out = self.res1(out)
        out = self.deconv1(out)

        out = torch.cat([out,input2],1)
        out = self.conv2(out)
        out = self.res2(out)
        out = self.deconv2(out)

        out = torch.cat([out,input3],1)
        out = self.conv3(out)
        out = self.res3(out)
        out = self.transition(out)

        return out