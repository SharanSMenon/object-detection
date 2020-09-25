import torch
import torch.nn as nn
import torch.nn.functional as F

###########################
# Prediction Convolutions #
###########################
class PredictionConvolutions(nn.Module):
    """Some Information about PredictionConvolutions"""
    def __init__(self):
        super(PredictionConvolutions, self).__init__()

    def forward(self, x):

        return x


#########################
# Auxilary Convolutions #
#########################
class AuxilaryConvolutions(nn.Module):
    """Some Information about AuxilaryConvolutions"""
    def __init__(self):
        super(AuxilaryConvolutions, self).__init__()

    def forward(self, x):

        return x