import torch
import torch.nn as nn
import torch.nn.functional as F
from ssd.models.vgg import VGGBase
from ssd.models.convolutions import PredictionConvolutions, AuxilaryConvolutions
from ssd.utils import *
from math import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#######
# SSD #
#######


class SSD(nn.Module):
    """Some Information about SSD"""

    def __init__(self, n_classes):
        super(SSD, self).__init__()
        self.vggbase = VGGBase()
        self.auxilary_convs = AuxilaryConvolutions()
        self.prediction_convs = PredictionConvolutions(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_anchor_boxes()

    def forward(self, x):

        conv4_3_feats, conv7_feats = self.vggbase(x)
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats /= norm
        conv4_3_feats *= self.rescale_factors

        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.auxilary_convs(
            conv7_feats)
        locs, classes_scores = self.prediction_convs(conv4_3_feats,
                                                     conv7_feats, conv8_2_feats,
                                                     conv9_2_feats, conv10_2_feats,
                                                     conv11_2_feats)

        return locs, classes_scores

    def detect_objects(self, predicted_locs, predicted_scores, min_score,
                       max_overlap, top_k):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  
                class_decoded_locs = decoded_locs[score_above_min_score]  

                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                overlap = jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    suppress = torch.max(suppress, overlap[box] > max_overlap)

                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k] 
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores

    def create_anchor_boxes(self):

        fmap_dims = {
            "conv4_3": 38,
            "conv7": 19,
            "conv8_2": 10,
            "conv9_2": 5,
            "conv10_2": 3,
            "conv11_2": 1
        }

        obj_scales = {
            "conv4_3": 0.1,
            "conv7": 0.2,
            "conv8_2": 0.375,
            "conv9_2": 0.55,
            "conv10_2": 0.725,
            "conv11_2": 0.9
        }

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())
        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append(
                            [cx, cy, obj_scales[fmap] * sqrt(ratio),
                             obj_scales[fmap] / sqrt(ratio)])

                        if ratio == 1:
                            try:
                                additional_scale = sqrt(
                                    obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append(
                                [cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)
        return prior_boxes
