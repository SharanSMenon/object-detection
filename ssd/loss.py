import torch
from torch import nn, optim

#################
# Multibox Loss #
#################


class MultiBoxLoss(nn.Module):
    """Implementation of Multibox loss"""

    def __init__(self, prior_cxcy, threshold=0.5, neg_pos_ratio=0.3, alpha=1.):
        super(MultiBoxLoss, self).__init__()

        self.priors_cxcy = prior_cxcy
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropy()

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
