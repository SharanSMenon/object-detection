import torch
import torch.nn as nn
import torch.nn.functional as F

#######
# SSD #
#######
class SSD(nn.Module):
    """Some Information about SSD"""
    def __init__(self):
        super(SSD, self).__init__()

    def forward(self, x):

        return x