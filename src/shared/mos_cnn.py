import torch
from typing import List
from torch import nn
from torch.nn import functional as F

from utils.utils import conv_same_padding


class MOSCNN(nn.Module):
    def __init__(self,
                 cnn_channels: List[int],
                 cnn_kernel_size: int
                 ):

        super(MOSCNN, self).__init__()
        self.cnn_kernel_size = cnn_kernel_size

        # for each element (x) in cnn_channels, adds 3 convolutional layers with x channels
        self.layers = nn.ModuleList([
            nn.Sequential(
                self.conv_layer(
                    in_channels=(1 if ind == 0 else cnn_channels[ind-1]),
                    out_channels=channels
                ),
                self.conv_layer(
                    in_channels=channels,
                    out_channels=channels
                ),
                self.conv_layer(
                    in_channels=channels,
                    out_channels=channels,
                    stride=(1, 3)
                )
            )
            for ind, channels in enumerate(cnn_channels)
        ])

    def conv_layer(self, in_channels: int, out_channels: int, stride: int = 1):
        # adds initialization function for conv layer and maintains same padding
        conv_layer = nn.Conv2d(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=self.cnn_kernel_size,
             stride=stride,
             padding=conv_same_padding(self.cnn_kernel_size)
        )
        nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='relu')
        return conv_layer

    def forward(self, x: torch.Tensor):
        #accepts a tensor [bz, frame_size, mel], assuming one channel
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return x