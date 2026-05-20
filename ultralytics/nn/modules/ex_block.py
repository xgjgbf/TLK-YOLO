import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import SPPF


class SPPFDS4(SPPF):
    def __init__(self, c1, c2, e=4):
        super().__init__(c1, c2)
        c_ = c1 // e  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  #创建实例,将输入通道数减少到隐藏通道数c_
        self.cv2 = Conv(c_ * 4 + c1, c2, 1, 1)  #创建另一个 Conv 层实例 cv2，它将输入通道数从 c_ * 3 变换到输出通道数 c2。
        # self.m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  #创建最大池化层 用padding 保持输出尺寸不变。
        # self.m2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.m1 = Conv(c_, c_, 3, 1, p=0, g=c_, d=1)
        self.m2 = Conv(c_, c_, 3, 1, p=0, g=c_, d=2)  # 创建最大池化层 用padding 保持输出尺寸不变。
        self.m3 = Conv(c_, c_, 3, 1, p=0, g=c_, d=3)

        # self.m1 = Conv(c_, c_, 3, 1)
        # self.m2 = Conv(c_, c_, 3, 1, d=2)
        # self.m3 = Conv(c_, c_, 3, 1, d=3)

    def forward(self, x):
        y0 = self.cv1(x)

        y1 = nn.functional.pad(y0, (1, 1, 1, 1), "constant", 0)  #pad值的计算：k//2   对于m1， d=1，3*3的特整图  k=3   3//2=1
        y1 = self.m1(y1)

        y2 = nn.functional.pad(y1, (2, 2, 2, 2), "constant", 0)  #对于m2，d=2，步长为2的填充，5*5特征图   k=5   5//2=2
        y2 = self.m2(y2)

        y3 = nn.functional.pad(y2, (3, 3, 3, 3), "constant", 0)  #对于m3，d=4，步长为4的填充，   ？？？？？？？
        y3 = self.m3(y3)

        return self.cv2(torch.cat((x, y0, y1, y2, y3), 1))  #拼接