import torch
from torch import nn, optim

#################
# Multibox Loss #
#################
class MultiBoxLoss(nn.Module):
    """Implementation of Multibox loss"""
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        
    def forward(self, x):
        return x