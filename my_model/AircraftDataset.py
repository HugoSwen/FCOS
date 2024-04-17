import os

import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class AircraftDataset(Dataset):
    # 类别
    CLASSES_NAME = (
        "__background__ ",
        "E2",
        "J20",
        "B2",
        "F14",
        "Tornado",
        "F4",
        "B52",
        "JAS39",
        "Mirage2000",
    )

    def __init__(self, data_path, mode, resize_size=[800, 1024]):
        self.img_path = None
        self.csv_path = None

        if mode != 'predict':
            # 图片路径
            if mode == 'train':
                dset = 'train_images'
            else:
                dset = 'val_images'
            self.img_path = os.path.join(data_path, dset, "%s.jpg")
            # csv路径
            self.csv_path = os.path.join(data_path, "csv", "%s.csv")
            # 获取每个图片的文件名
            image_ids = os.listdir(os.path.join(data_path, dset))
            self.img_ids = [os.path.splitext(img_id)[0] for img_id in image_ids]
        else:
            self.img_ids = [os.path.splitext(data_path)[0]]
        # 类别与序号映射
        self.classMap = dict(zip(AircraftDataset.CLASSES_NAME, range(len(AircraftDataset.CLASSES_NAME))))
        # 图片归一化参数
        self.resize_size = resize_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print("AircraftDataset init finished...")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        if self.img_path is not None:
            # 获取每张图片的rgb图片
            img = self._read_img_rgb(self.img_path % img_id)
        else:
            img = self._read_img_rgb(img_id + '.jpg')
        if self.csv_path is not None:
            # 获取每个目标的坐标和类别
            df = pd.read_csv(self.csv_path % img_id)
        else:
            df = pd.DataFrame()
        boxes = []
        classes = []
        for i in range(len(df)):
            box = [
                df.loc[i, 'xmin'],
                df.loc[i, 'ymin'],
                df.loc[i, 'xmax'],
                df.loc[i, 'ymax'],
            ]

            TO_REMOVE = 1
            box = [float(coord - TO_REMOVE) for coord in box]
            boxes.append(box)

            className = self.classMap[df.loc[i, 'class']]
            classes.append(className)

        boxes = np.array(boxes, dtype=np.float32)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    @staticmethod
    def _read_img_rgb(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    @staticmethod
    def preprocess_img_boxes(image, boxes, input_ksize):
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 0 if 32 - nw % 32 == 32 else 32 - nw % 32
        pad_h = 0 if 32 - nh % 32 == 32 else 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None or boxes.size == 0:
            return image_paded, boxes
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num: max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes


if __name__ == "__main__":
    import cv2

    dataset = AircraftDataset("../dataset", mode='train', resize_size=[512, 800])
    imgs, boxes, classes = dataset.collate_fn([dataset[20], dataset[40], dataset[50]])
    print(boxes, classes, "\n", imgs.shape, boxes.shape, classes.shape, boxes.dtype, classes.dtype, imgs.dtype)

    for index, i in enumerate(imgs):
        i = i.numpy().astype(np.uint8)
        i = np.transpose(i, (1, 2, 0))
        i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        print(i.shape, type(i))
        cv2.imwrite(str(index) + ".jpg", i)
