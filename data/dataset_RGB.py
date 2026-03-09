import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderFileTrain(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderFileTrain, self).__init__()

        self.inp_filenames = os.listdir(rgb_dir)
        self.inp_filenames = [filename for filename in self.inp_filenames if filename.endswith('0.png')]

        self.ir_dir = rgb_dir
        self.sizex       = len(self.inp_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        # ps = self.ps

        inp_path = self.inp_filenames[index_]
        vi_path = inp_path

        inp_img = os.path.join(self.ir_dir, inp_path)
        tar_img = os.path.join(self.ir_dir.replace('ir','vi'), vi_path)

        ir = self.imread(path=inp_img, flags='L')
        vi = self.imread(path=tar_img, flags='RGB')

        vi = (vi - torch.min(vi)) / (torch.max(vi) - torch.min(vi))
        ir = (ir - torch.min(ir)) / (torch.max(ir) - torch.min(ir))

        return ir, vi, inp_path, 0

    @staticmethod
    def imread(path, flags='RGB'):
        im = Image.open(path).convert(flags)
        im = TF.to_tensor(im)
        return im


class DataLoaderFile1(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderFile1, self).__init__()

        self.inp_filenames = os.listdir(rgb_dir)
        self.inp_filenames = [filename for filename in self.inp_filenames if filename.endswith('1.png')]

        self.ir_dir = rgb_dir
        self.sizex       = len(self.inp_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        # ps = self.ps

        inp_path = self.inp_filenames[index_]
        vi_path = inp_path

        inp_img = os.path.join(self.ir_dir, inp_path)
        tar_img = os.path.join(self.ir_dir.replace('ir','vi'), vi_path)

        ir = self.imread(path=inp_img, flags='L')
        vi = self.imread(path=tar_img, flags='RGB')
        # gt = self.imread(path=gt_path, flags='RGB')

        vi = (vi - torch.min(vi)) / (torch.max(vi) - torch.min(vi))
        ir = (ir - torch.min(ir)) / (torch.max(ir) - torch.min(ir))
        # gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
        return ir, vi, inp_path, 1


    @staticmethod
    def imread(path, flags='RGB'):
        im = Image.open(path).convert(flags)
        im = TF.to_tensor(im)
        return im


class DataLoaderFileVal(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderFileVal, self).__init__()

        self.inp_filenames = os.listdir(rgb_dir)
        self.ir_dir = rgb_dir
        self.sizex       = len(self.inp_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        # ps = self.ps

        inp_path = self.inp_filenames[index_]
        vi_path = inp_path

        ir_img = os.path.join(self.ir_dir, inp_path)
        vi_img = os.path.join(self.ir_dir.replace('ir','vi'), vi_path)
        ir = self.imread(path=ir_img, flags='L')
        vi = self.imread(path=vi_img, flags='RGB')


        vi = (vi - torch.min(vi)) / (torch.max(vi) - torch.min(vi))
        ir = (ir - torch.min(ir)) / (torch.max(ir) - torch.min(ir))

        return ir, vi, inp_path
    @staticmethod
    def imread(path, flags='RGB'):
        im = Image.open(path).convert(flags)
        im = TF.to_tensor(im)
        return im


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


class DataLoaderFileTest(Dataset):
    def __init__(self, ir_dir):
        super(DataLoaderFileTest, self).__init__()

        self.input_names = os.listdir(ir_dir)
        self.inp_size = len(self.input_names)
        self.input_dir = ir_dir

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        inp_path = self.input_names[index]
        vi_path = inp_path
        inp_img = os.path.join(self.input_dir, inp_path)

        vi_img = os.path.join(self.input_dir.replace('ir','vi'), vi_path)

        ir = self.imread(path=inp_img, flags='L')
        vi = self.imread(path=vi_img, flags='RGB')


        vi = (vi - torch.min(vi)) / (torch.max(vi) - torch.min(vi))
        ir = (ir - torch.min(ir)) / (torch.max(ir) - torch.min(ir))
        return ir, vi, inp_path
    @staticmethod
    def imread(path, flags='RGB'):
        im = Image.open(path).convert(flags)
        im = TF.to_tensor(im)
        return im