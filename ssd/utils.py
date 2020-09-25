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

def xy_to_center():
    pass

def center_to_xy():
    pass