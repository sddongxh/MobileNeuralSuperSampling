import torch
import torch.nn as nn
import time
# time_start = time.time()  # 开始计时
# time_end = time.time()  # 结束计时
# time_c = time_end - time_start  # 运行所花时间
# print('dataload 时间：', time_c, 's')
from torchvision.transforms import ToPILImage
show = ToPILImage()
flag = False
path = r"C:/Users/AAAA/Desktop/res/"


class MNSS(nn.Module):
    def __init__(self):
        super(MNSS, self).__init__()
        self.Weight = HandWeight()
        self.Merging = Merging()

    def forward(self, x_RGBD_upsampling, previous_RGBD_Warped):
        #0 upsampling of current view

        #1 warping of previous view and depth

        #2 calculate the weight and multiply previous view
        weight = self.Weight(x_RGBD_upsampling, previous_RGBD_Warped)
        weighted_pre_RGBD = torch.mul(previous_RGBD_Warped, weight)

        #3 Merging

        output = self.Merging(torch.cat((x_RGBD_upsampling, weighted_pre_RGBD), 1))
        return output


class HandWeight(nn.Module):
    def __init__(self, num_previous=4):
        super(HandWeight, self).__init__()
        kernel_size = 3
        padding = 1

        Weight = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=(kernel_size, kernel_size), padding=padding, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=(kernel_size, kernel_size), padding=padding, padding_mode='replicate'),
            nn.Tanh(),
        )
        self.add_module("Weight", Weight)

    def forward(self, RGBD: torch.Tensor, PreRGBD: torch.Tensor) -> torch.Tensor:
        x = torch.abs(RGBD-PreRGBD)
        out = self.Weight(x)
        return out


class Merging(nn.Module):
    def __init__(self):
        super(Merging, self).__init__()
        Dilation_CNN1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=5, dilation=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        Dilation_CNN1_2 = nn.Sequential(
            nn.Conv2d(8+8, 8, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.add_module("Dilation_CNN1", Dilation_CNN1)
        self.add_module("Dilation_CNN1_2", Dilation_CNN1_2)


    def forward(self, mergRGBD: torch.Tensor) -> torch.Tensor:


        Dilation_out1 = self.Dilation_CNN1(mergRGBD)

        OneD_out2 = self.Dilation_CNN1_2(torch.cat((mergRGBD, Dilation_out1), dim=1))
        return OneD_out2
