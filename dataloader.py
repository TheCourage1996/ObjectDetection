# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import cv2
import random
class VOCDataset(Dataset):
    def __init__(self, root, split='train',num_classes=21, base_size=None, augment=True,
                 crop_size=321, scale=True, flip=True, rotate=True, blur=True,):
        super(VOCDataset, self).__init__()
        self.root = root     # 存放数据集的根路径
        self.num_classes = num_classes  # 数据集的类别总数
        self.MEAN = [0.45734706, 0.43338275, 0.40058118] # 数据集的均值和方差
        self.STD = [0.23965294, 0.23532275, 0.2398498]
        self.crop_size = crop_size  #裁剪图片的大小
        self.scale = scale          #是否进行scale
        self.flip = flip            #是否进行flip
        self.rotate = rotate        #是否进行rotate
        self.blur = blur            # 是否进行blur
        self.base_size = base_size  # 基础读入图片大小
        self.augment = augment  #是否进行数据增强
        self.split = split  # 拿到训练模式
        self._set_files()   # 调用函数，拿到所有训练 验证的图片名字
        self.to_tensor = transforms.ToTensor() # 对图片进行归一化处理
        self.normalize = transforms.Normalize(self.MEAN,self.STD)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOC2012')  # VOC数据集的路径
        self.image_dir = os.path.join(self.root, 'JPEGImages') # 图片的存放路径
        self.label_dir = os.path.join(self.root, 'SegmentationClass') # 标签的存放路径
        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        # 训练或验证图片的名称txt文件
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))] # 训练或验证图片的名称 放入列表
        # 这里拿到的是对应的图片的名字  放在列表中

    def _load_data(self, index):
        image_id = self.files[index]  # 根据索引取图片
        image_path = os.path.join(self.image_dir, image_id + '.jpg') # 图片路径
        label_path = os.path.join(self.label_dir, image_id + '.png') # 标签路径
        # 将图片转成数组
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label

    def __getitem__(self, index):
        "__getitem__方法在自定义数据集的时候必须重写.index是输入图片的索引值"
        "在这个函数里面可以对图片进行预处理，但是要返回处理好的图片"
        image, label = self._load_data(index)   # 拿到每一张图片和标签
        if self.augment: # 判断是否进行数据增强
            image, label = self._augmentation(image, label)
        # 统一输入图片格式
        label = torch.from_numpy(np.array(label, dtype=np.float32)).long()
        image = Image.fromarray(np.uint8(image))
        return self.normalize(self.to_tensor(image)), label # 归一化 将图片转换为tensor对象

    def __len__(self):
        "__len__方法在自定义数据集时候必须重写.返回数据集的长度"
        return len(self.files)

    #数据增强函数
    def _augmentation(self, image, label):
        h, w, _ = image.shape
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (
            int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape
        # 旋转图片在（-10°和10°之间）
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, rot_matrix, (w, h),
                                   flags=cv2.INTER_NEAREST)

        # 对不符合指定大小的图片进行裁剪
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT, }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

            # 对不符合大小的图片进行padding
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # 随机反转
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # 给图片增加高斯噪音
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT_101)
        return image, label




