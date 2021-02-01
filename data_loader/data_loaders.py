from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np
import torch
from PIL import Image
import os
import ntpath
import matplotlib.pyplot as plt

def read_train_dataset(root, unknown_class=11):
    images = []
    labels = []

    with open(os.path.join(root, 'train_labels.txt'), 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if len(line):
                path = os.path.join('bars', ' '.join(line.split()[:-1]))
                strlabel = line.split()[-1]
                # if len(strlabel) == 5 and '-' not in strlabel:
                #     label = ','.join(list(strlabel))
                # else:
                    # pass
                arrlabel = [l if l != '-' else str(unknown_class) for l in list(strlabel)]
                y = np.full(8, unknown_class).astype(np.str)
                y[0:len(arrlabel)] = arrlabel
                label = ','.join(y)
                images.append(path)
                labels.append(label)

    # for t in ['train.txt', 'test.txt']:
    #     txt_fname = os.path.join(root,'difficult_samples_for_'+t)
    #     with open(txt_fname, 'r') as f:
    #         for line in f.readlines():
    #             label = line.split()[1]
    #             arrlabel = np.array(label.split(',')).astype(np.int)
    #             if arrlabel.max() <= 9:
    #                 y = np.full(8, unknown_class).astype(np.str)
    #                 y[0:len(arrlabel)] = arrlabel
    #                 label = ','.join(y)
    #                 labels.append(label)
    #                 images.append(line.split()[0])
    #         f.close()

    # txt_fname_easy = os.path.join(root, 'easy_samples.txt')
    # with open(txt_fname_easy, 'r') as f:
    #     for line in f.readlines():
    #         label = line.split()[1]
    #         arrlabel = np.array(label.split(',')).astype(np.int)
    #         if arrlabel.max() <= 9:
    #             y = np.full(8, unknown_class).astype(np.str)
    #             y[0:len(arrlabel)] = arrlabel
    #             label = ','.join(y)
    #             labels.append(label)
    #             images.append(line.split()[0])
    #     f.close()

    return images, labels

def read_test_dataset(root, unknown_class=11):
    images = []
    labels = []
    with open(os.path.join(root, 'test_labels.txt'), 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            if len(line):
                path = os.path.join('bars', ' '.join(line.split()[:-1]))
                strlabel = line.split()[-1]
                # if len(strlabel) == 5 and '-' not in strlabel:
                #     label = ','.join(list(strlabel))
                # else:
                # pass
                arrlabel = [l if l != '-' else str(unknown_class) for l in list(strlabel)]
                y = np.full(8, unknown_class).astype(np.str)
                y[0:len(arrlabel)] = arrlabel
                label = ','.join(y)

                images.append(path)
                labels.append(label)
    return images, labels

def read_images_from_folder(root):
    images = []
    for img_path in os.listdir(root):
        images.append(ntpath.basename(img_path))
    return images

class WaterMeterDataset(Dataset):

    unknown_class = 11

    def __init__(self, root, mode='train' ):

        assert(mode in {'train', 'test', 'predict'})

        self.mode = mode
        self.root = root

        if self.mode == 'train':
            self.data, self.label = read_train_dataset(root, self.unknown_class)
        elif self.mode == 'test':
            self.data, self.label = read_test_dataset(root, self.unknown_class)
        elif self.mode == 'predict':
            self.data = read_images_from_folder(root)
            self.label = None

    def __getitem__(self, index):
        img = self.data[index]
        img_path = img
        img = Image.open(os.path.join(self.root, img))
        # 根据不同图片的大小进行缩放和补零操作，使得水表图像具有相同的高度H和宽度W
        H, W = 48, 160
        ratio = img.height/img.width
        if self.mode == 'train':
            augs = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomAffine(degrees=(-5,5), scale=(.9, 1.1), shear=0),
                transforms.RandomPerspective(distortion_scale=0.08),
                # transforms.RandomErasing(p=0.15, scale=(0.02, 0.16), ratio=(0.05, 0.1), value=0),
            ])
        else:
            augs = transforms.Compose([])

        h_ = (H-W*ratio)/2
        self.transforms = transforms.Compose([
            transforms.Resize((int(W*ratio), W)),
            augs,
            transforms.Pad((0, int(h_))),
            transforms.Resize((H, W)),  # 防止每张图片缩放补零后大小不一致
            transforms.ToTensor(),
        ])

        img = self.transforms(img)
        
        # plt.imsave(os.path.join('predictions', img_path), np.transpose(img.cpu().numpy(), (1,2,0)))

        if self.mode in {'train', 'test'}:
            label = self.label[index]  #获得字符串'0,1,2,5,14'
            label = label.split(',')  #获得列表['0','1','2','5','14']
            label = [int(i) for i in label]  # 转换为[0, 1, 2, 5, 4]
            label = torch.from_numpy(np.array(label)).long()

            return img, label
        else:
            return img, img_path

    def __len__(self):
        return len(self.data)


class WaterMeterDataLoader(BaseDataLoader):
    """
    WaterMeter loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, mode='train'):
        self.data_dir = data_dir
        self.dataset = WaterMeterDataset(self.data_dir, mode=mode)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
