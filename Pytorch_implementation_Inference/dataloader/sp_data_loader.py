import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tf
from PIL import Image
import json
import time
import random
from utils.util import unpackage, unpackage_depth


class Mutil_IMG2IMG(Dataset):
    def __init__(self,
                 root_dir: str,
                 batch_size: int,
                 ):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.sceneslist = os.listdir(self.root_dir)
        random.shuffle(self.sceneslist)
        self.sceneslist = self.sceneslist[0:self.batch_size]
        print(self.sceneslist)
        img = os.listdir(os.path.join(self.root_dir, self.sceneslist[0], 'CBR_3x'))
        img.sort(key=lambda x: int(x[:-4]))
        self.imglist = [i for i in img for j in range(self.batch_size)]
        self.transform = tf.ToTensor()

    def __getitem__(self, index):
        scenesid = index % self.batch_size
        scene = self.sceneslist[scenesid]
        image_name = self.imglist[index]

        view_path = os.path.join(self.root_dir, scene, 'CBR_3x', image_name)
        depth_path = os.path.join(self.root_dir, scene, 'CBR_3d', image_name)
        flow_pathx = os.path.join(self.root_dir, scene, 'motion_vectors_x_3x', image_name)
        flow_pathy = os.path.join(self.root_dir, scene, 'motion_vectors_y_3x', image_name)

        img = Image.open(view_path)
        img_view = self.transform(img)
        img_view = img_view[0:3]

        img_depth = self.transform(Image.open(depth_path))
        img_depth = unpackage_depth(img_depth)

        img_flowx = self.transform(Image.open(flow_pathx))
        img_flowy = self.transform(Image.open(flow_pathy))
        img_flow = unpackage(img_flowx, img_flowy)

        GT_img_view = []

        return img_view, img_depth, img_flow, GT_img_view, image_name

    def __len__(self) -> int:
        return len(self.imglist)

