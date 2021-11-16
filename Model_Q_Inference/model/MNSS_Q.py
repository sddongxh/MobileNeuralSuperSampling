import torch
import torch.nn as nn

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
        output = torch.clip(output, 0.0, 1.0)
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
        OneD_CNN1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        Dilation_CNN1 = nn.Sequential(
            nn.Conv2d(8+8, 16, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=5, dilation=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        Dilation_CNN2 = nn.Sequential(
            nn.Conv2d(8+8+8, 16, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=5, dilation=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        Dilation_CNN3 = nn.Sequential(
            nn.Conv2d(8+8+8+8, 16, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=5, dilation=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.add_module("OneD_CNN1", OneD_CNN1)
        self.add_module("Dilation_CNN1", Dilation_CNN1)
        self.add_module("Dilation_CNN2", Dilation_CNN2)
        self.add_module("Dilation_CNN3", Dilation_CNN3)

    def forward(self, mergRGBD: torch.Tensor) -> torch.Tensor:
        OneD_out1 = self.OneD_CNN1(mergRGBD)
        Dilation_out1 = self.Dilation_CNN1(torch.cat((mergRGBD, OneD_out1), dim=1))
        OneD_out2 = self.Dilation_CNN2(torch.cat((mergRGBD, OneD_out1, Dilation_out1), dim=1))
        OneD_out3 = self.Dilation_CNN3(torch.cat((mergRGBD, OneD_out1, Dilation_out1, OneD_out2), dim=1))
        return OneD_out3
