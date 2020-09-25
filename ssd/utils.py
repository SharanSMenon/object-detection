import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as FT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
voc_labels: list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

label_map: dict = dict(zip(voc_labels, list(range(1, len(voc_labels) + 1))))
label_map["background"] = 0
rev_label_map: dict = {v: k for k, v in label_map.items()}

distinct_colors: list = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map: dict = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

###################
# Transformations #
###################

def expand(image: torch.Tensor, boxes, filler):
    orig_w = image.size(1)
    orig_h = image.size(2)
    max_scale=4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * orig_h)
    new_w = int(scale * orig_w)
    filler = torch.FloatTensor(filler)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)

    left = random.randint(0, new_w - orig_w)
    right = left + orig_w
    top = random.randing(0, new_h - orig_h)
    bottom = top + orig_h
    new_image[:, top:bottom, left:right] = image
    new_boxes = boxes + torch.FloatTensor([left, right, top, bottom]).unsqueeze(0)
    return new_image, new_boxes

def resize(image, boxes, percent=True):
    new_image = FT.resize(image, (300, 300))

    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims
    if not percent:
        new_dims = torch.FloatTensor([300, 300, 300, 300]).unsqueeze(0)
        new_boxes = new_boxes * new_dims
    return new_image, new_boxes

def distort(image):
    distortions: list = [
        FT.adjust_brightness,
        FT.adjust_hue,
        FT.adjust_contrast,
        FT.adjust_saturation
    ]
    for d in distortions:
        if random.random() < 0.45:
            if d.__name__ == "adjust_hue":
                adjust_factor = random.uniform(-18/255, 18/255)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
            new_img = d(image, adjust_factor)
    return new_img

def flip(image, boxes):
    new_image = FT.hflip(image)

    new_b = boxes
    new_b[:, 0] = image.width - boxes[:, 0] - 1
    new_b[:, 2] = image.height - boxes[:, 2] - 1
    new_b = new_b[:, [2, 1, 0, 3]]
    return new_image, new_b

def transform(image, boxes, labels, difficulties, split) -> tuple:
    """
    Does augmentations and transforms the Bounding Boxes to any required specificiation.

    Args:
        image (PIL.Image): The Image
        boxes (list): Bounding boxes for the object
        labels (list): Class Labels for the Objects
        difficulties (list): Whether the object is difficult or not
        split (str): Train or test

    Returns:
        tuple: (new_image, new_bounding_boxes, labels, difficulties)
    """
    ### Mean and STD from VGG Paper ###
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes

    if split == 'TRAIN':
        new_image = distort(image)
        if random.random() > 0.5:
            new_image, new_boxes = flip(image, boxes)
        new_image = FT.to_tensor(new_image) 
        if random.random() > 0.5:
            new_image, new_boxes = expand(image, boxes, mean)
    
    new_image = FT.to_pil_image(new_image)
    new_image, new_boxes = resize(new_image, new_boxes)
    new_image = FT.to_tensor(new_image)
    new_image = FT.normalize(new_image, mean=mean, std=mean)

    return (new_image, new_boxes, labels, difficulties)

##########################
# Bounding Box Utilities #
##########################

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h