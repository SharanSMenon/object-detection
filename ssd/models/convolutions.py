import torch
import torch.nn as nn
import torch.nn.functional as F

###########################
# Prediction Convolutions #
###########################


class PredictionConvolutions(nn.Module):
    """Some Information about PredictionConvolutions"""

    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {
            "conv4_3": 4,
            "conv7": 6,
            "conv8_2": 6,
            "conv9_2": 6,
            "conv10_2": 4,
            "conv11_2": 4
        }

        self.loc_conv4_3 = nn.Conv2d(
            256, n_boxes['conv4_3']*4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(
            256, n_boxes['conv8']*4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(
            256, n_boxes['conv8_2']*4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(
            256, n_boxes['conv9_2']*4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(
            256, n_boxes['conv10_2']*4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(
            256, n_boxes['conv11_2']*4, kernel_size=3, padding=1)

        self.cl_conv4_3 = nn.Conv2d(
            256, n_boxes['conv4_3']*n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(
            256, n_boxes['conv7']*n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(
            256, n_boxes['conv8_2']*n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(
            256, n_boxes['conv9_2']*n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(
            256, n_boxes['conv10_2']*n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(
            256, n_boxes['conv11_2']*n_classes, kernel_size=3, padding=1)

        self.init_conv()

    def init_conv(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats,
                conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):

        batch_size = conv4_3_feats.size(0)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # 5776 boxes

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # 2166 boxes

        l_conv8_2 = self.loc_conv4_3(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # 600 boxes

        l_conv9_2 = self.loc_conv4_3(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # 150 boxes

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # 36 boxes

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # 4 boxes

        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv4_3(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8_2 = self.cl_conv4_3(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv9_2 = self.cl_conv4_3(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv10_2 = self.cl_conv4_3(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        c_conv11_2 = self.cl_conv4_3(conv8_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2,
                          l_conv10_2, l_conv11_2],
                         dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2,
                                    c_conv9_2, c_conv10_2, c_conv11_2],
                                   dim=1)

        return locs, classes_scores


#########################
# Auxilary Convolutions #
#########################
class AuxilaryConvolutions(nn.Module):
    """Takes output from VGG Base."""

    def __init__(self):
        super(AuxilaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=2, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=2, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=2, stride=2)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=2, stride=2)

        self.init_conv()

    def init_conv(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, x):  # x = conv_7_feats
        out = F.relu(self.conv8_1(x))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(x))  # (N, 512, 10, 10)
        conv8_2_feats = out

        out = F.relu(self.conv9_1(x))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(x))  # (N, 256, 5, 5)
        conv9_2_feats = out

        out = F.relu(self.conv10_1(x))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(x))  # (N, 256, 3, 3)
        conv10_2_feats = out

        out = F.relu(self.conv11_1(x))  # (N, 128, 3, 3)
        out = F.relu(self.conv11_2(x))  # (N, 256, 1, 1)
        conv11_2_feats = out

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats
