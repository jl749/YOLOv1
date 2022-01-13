"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""
from typing import List

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"maxpool" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3),
    "maxpool",
    (3, 192, 1, 1),
    "maxpool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "maxpool",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "maxpool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class Conv(nn.Module):
    def __init__(self, c1, c2, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, c1=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.c1 = c1
        self.darknet = self._create_architecture(self.architecture)
        self.fcs = self._create_decodeLayer(**kwargs)

    def forward(self, x):
        x = self.darknet(x)  # convolution layer
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_architecture(self, architecture):
        layers: List[nn.Module] = []  # 1d array
        c1 = self.c1

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    Conv(
                        c1, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                c1 = x[1]  # current c2 becomes next c1

            elif type(x) == str:
                if x == "maxpool":
                    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                iter_count = x[2]

                for _ in range(iter_count):
                    layers += [
                        Conv(
                            c1,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        Conv(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    c1 = conv2[1]

        return nn.Sequential(*layers)

    def _create_decodeLayer(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.SiLU(),
            # BoundingBox_info = [prob, centreCorX, centreCorY, boxWidth, boxHeight]
            nn.Linear(4096, S * S * (C + B * 5)),  # later reshape into (S, S, 30)  hence, C+B*5 == 30
        )


# class MyYolov1(nn.Module):
#     def __init__(self, c1=3, batch_size=1):
#         super(MyYolov1, self).__init__()
#         self.c1 = c1
#         self.batch_size = batch_size
#
#         self.conv1 = Conv(c1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
#         self.mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
#         self.conv2 = Conv(64, 192, kernel_size=(3, 3), stride=1, padding=1)
#         self.mp2 = nn.MaxPool2d((2, 2), 2)
#
#         self.conv3 = Conv(192, 128, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv4 = Conv(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.conv5 = Conv(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv6 = Conv(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.mp3 = nn.MaxPool2d((2, 2), 2)
#
#         self.conv7 = Conv(512, 256, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv8 = Conv(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.conv9 = Conv(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv10 = Conv(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.mp4 = nn.MaxPool2d((2, 2), 2)
#
#         self.conv11 = Conv(1024, 512, kernel_size=(1, 1), stride=(1, 1), padding=0)
#         self.conv12 = Conv(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.conv13 = Conv(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.conv14 = Conv(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.conv15 = Conv(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1)
#         self.conv16 = Conv(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1)
#
#         self.linear1 = nn.Linear(1024, 4096)
#         self.linear2 = nn.Linear(4096, 30)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.mp1(x)
#
#         x = self.conv2(x)
#         x = self.mp2(x)
#
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.mp3(x)
#
#         for _ in range(4):
#             x = self.conv7(x)
#             x = self.conv8(x)
#         x = self.conv9(x)
#         x = self.conv10(x)
#         x = self.mp4(x)
#
#         for _ in range(2):
#             x = self.conv11(x)
#             x = self.conv12(x)
#         x = self.conv13(x)
#         x = self.conv14(x)
#         x = self.conv15(x)
#         x = self.conv16(x)
#
#         split_size = x.shape[2]
#
#         x = torch.flatten(x, start_dim=1)  # x = nn.Flatten()(x)
#         x = nn.Linear(x.shape[1], 4096)(x)
#         x = nn.Linear(x.shape[1], 30*split_size**2)(x)
#
#         return x.reshape(self.batch_size, split_size, split_size, 30)
#
#
# def test_Yolov1(split_size=7, num_boxes=2, num_classes=20):  # 2*5 + 20 = 30
#     model = Yolov1(split_size, num_boxes, num_classes)
#     x = torch.randn((2, 448, 448, 3))
#     model(x)
