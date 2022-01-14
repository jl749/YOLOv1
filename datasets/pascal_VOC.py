"""
Creates a Pytorch datasets to load the Pascal VOC datasets
"""

import torch
import os
import pandas as pd
from PIL import Image
from pathlib import Path


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file: Path, img_dir: Path, label_dir: Path,
                 split_size=7, num_boxes=2, num_classes=20, transform=None  # for data augmentation if wanted
                 ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):  # index = which row in csv
        """
        Get input index(image) corresponding label info
        :param index: which row in csv (image index)
        :return:
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])  # data/labels/*.txt
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])
        # boxes = [[class_label, x, y, width, height], ... max 2]
        boxes = torch.tensor(boxes)

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))  # 7, 7, 20 + 2*5
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            cell_x, cell_y = self.S * x - j, self.S * y - i  # x,y coordinate within the cell

            cell_w, cell_h = (  # scale up w,h accordingly (0~1) --> (0~S)
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object per cell!
            if label_matrix[i, j, 20] == 0:
                # set that there exists an object
                label_matrix[i, j, 20] = 1

                # box coordinates
                box_coordinates = torch.tensor(
                    [cell_x, cell_y, cell_w, cell_h]
                )

                # set x,y,w,h
                label_matrix[i, j, 21:25] = box_coordinates
                # set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
