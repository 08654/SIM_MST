from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as   F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from torch.backends import cudnn
from operator import truediv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import random
import torch
from tqdm import tqdm
import sys
import os
from scipy.io import loadmat

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class TDCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(TDCNN, self).__init__()
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1),
            nn.BatchNorm3d(8),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout(0.3),
        )
        # self.cbam1 = CBAM(8)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1), dilation=2),
            nn.BatchNorm3d(16),
            nn.Dropout(0.3)
        )
        # self.cbam2 = CBAM(16)

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1),
            nn.BatchNorm3d(32)
        )

        # Additional Conv Layers
        self.conv8 = nn.Conv3d(32, 64, kernel_size=(1, 1, 1))
        self.conv16 = nn.Conv3d(64, 128, kernel_size=(1, 1, 1))
        self.conv32 = nn.Conv3d(128, 256, kernel_size=(1, 1, 1))
        self.conv32_res = nn.Conv3d(32, 256, kernel_size=(1, 1, 1))
        self.conv32_res1 = nn.Conv3d(256, 128, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        # Fully Connected Layers
        self.linear1 = nn.Sequential(
            nn.Linear(18432, 2048),  # Adjusted based on Flatten output
            nn.Dropout(0.3)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.3)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(1024, 256)
        )
        self.linear4 = nn.Linear(256, num_classes)

    def forward(self, input):
        # Layer 1
        input = self.layer1(input)
        # input = F.relu(self.cbam1(input))

        # Layer 2
        input = self.layer2(input)
        # input = F.relu(self.cbam2(input))

        # Layer 3
        input = self.layer3(input)
        input = F.relu(input)

        # Additional Conv Layers
        input = self.conv8(input)
        input = self.conv16(input)
        input = self.conv32(input)
        input = self.conv32_res1(input)

        # Flatten and Fully Connected Layers
        input = input.flatten(1)
        input = F.relu(self.linear1(input))
        input = F.relu(self.linear2(input))
        input = F.relu(self.linear3(input))
        output = self.linear4(input)
        return output
#定义
def TD_CNN_model():
    return TDCNN(num_classes=1)