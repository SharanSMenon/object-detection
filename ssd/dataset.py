import torch.utils.data as data
import xml.etree.ElementTree as ET
import torch
from PIL import Image
import collections
from torchvision import datasets, transforms
import os
from utils import transform

VOC_CLASSES: tuple = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCDetection(datasets.VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, 
            ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entryand returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            transform=None,
            target_transform=None,
            transforms=None,
    ):
        super(VOCDetection, self).__init__(
            root, transforms, transform, target_transform)
        self.year = year
        voc_root = os.path.join(self.root, "VOC2012")
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if image_set == "train":
            self.train = True
        else:
            self.train = False

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(
            annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def transform_annotation_to_bbox(self, target):
        """
        Transforms annotations 

        Args:
            target ([type]): [description]
        """
        class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

        res = []
        classes = []
        difficulties = []
        for obj in target["annotation"]["object"]:
            difficult = int(obj['difficult']) == 1
            name = obj["name"].lower().strip()
            bbox = obj['bndbox']
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox[pt]) - 1
#                 cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = class_to_ind[name]
            classes.append(label_idx)
            difficulties.append(difficult)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res, classes, difficulties

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        img = trans(img)

        bndboxes, clses, difficulties = self.transform_annotation_to_bbox(
            target)

        image, bboxes, labels, difficulties = transform(img, bndboxes, clses,
                                                        difficulties, 
                                                        "TRAIN" if self.train else "TEST")

        return image, bboxes, labels, difficulties

    def __len__(self) -> int:
        return len(self.images)

    def parse_voc_xml(self, node):
        """
        Parses the XML Annotations

        Args:
            node ([type]): The XML Node

        Returns:
            [type]: A dictionary containing the parsed annotation
        """
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def collate_fn(self, batch):
        """
        Collate Function for the PyTorch Dataloader

        Args:
            batch (list): Batch of images and its objects
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties
