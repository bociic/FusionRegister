import os
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from .common import RGB2YCrCb
from torchvision import transforms

class Generation:
    def __init__(self, config, args=None):
        self.config = config
        self.args = args
        self.transforms = transforms.ToTensor()

    def get_loaders(self, parse_patches=True, validation='Generation'):
        print("=> Utilizing the GenerationDataset() for data loading...")
        train_dataset = GenerationDataset(dir=os.path.join(self.config.data.data_dir, 'MSRS/train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.patch_size,                           
                                        transforms=self.transforms,
                                        scale = self.config.data.scale,
                                        parse_patches=parse_patches,
                                        phase='train')
        
        val_dataset = GenerationDataset(dir=os.path.join(self.config.data.data_dir, 'MSRS/test'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.patch_size,
                                      transforms=self.transforms,
                                      parse_patches=parse_patches,
                                      phase='val')


        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
    
    def get_test_loaders(self, parse_patches=True, phase='test', data_dir=None):
        print("=> Utilizing the GenerationDataset() for data loading...")
        # if data_dir is None:
        #     data_dir = os.path.join(self.config.data.data_dir, 'MSRS/test')
        # else:
        #     data_dir = os.path.join(data_dir)
        data_dir =  os.path.join(self.config.data.data_dir)
        test_dataset = GenerationDataset(
            dir=data_dir,
            n=self.config.training.patch_n,
            patch_size=self.config.data.patch_size,  
            transforms=self.transforms,         
            parse_patches=parse_patches,
            phase=phase                             
        )

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.sampling.batch_size,   
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return test_loader


class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, scale=1,parse_patches=True, phase='train'):
        super().__init__()
        print('source dir: ', dir)
        self.phase = phase
        source_dir = dir
        vi = os.path.join(source_dir, 'vi')
        ir = os.path.join(source_dir, 'ir')
        print("vi floder: {}, ir folder: {}".format(vi, ir))
        vis, irs = [], []
        file_list = os.listdir(ir)
        file_list.sort()
        for item in file_list:                
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.bmp'):
                vis.append(os.path.join(vi, item))
                irs.append(os.path.join(ir, item))
        print("The number of the training dataset is: {}".format(len(irs)))
        
        if self.phase == 'train':
            x = list(enumerate(vis))
            random.shuffle(x)
            indices, vis = zip(*x)
            irs = [irs[idx] for idx in indices]
        
        self.dir = None        
        print("The number of the testing dataset is: {}".format(len(irs)))
        self.vis = vis
        self.irs = irs
        self.patch_size = patch_size
        self.scale = scale
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)] 
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        vi = self.vis[index]
        ir = self.irs[index]
        img_id = re.split('/', vi)[-1]
        vi = PIL.Image.open(vi)
        ir = PIL.Image.open(ir).convert('L')
        if self.phase == 'train':
            original_size = vi.size  
            new_size = (int(original_size[0] * self.scale), int(original_size[1] * self.scale))
            vi = vi.resize(new_size, PIL.Image.Resampling.LANCZOS)  
            ir = ir.resize(new_size, PIL.Image.Resampling.LANCZOS)

            vi_np = np.array(vi)  # shape: (H,W,3)
            ir_np = np.array(ir)[..., np.newaxis]  # shape: (H,W,1)
            img = np.concatenate([vi_np,ir_np],axis=2)
            imgs = self.transforms(img)  
            vi, ir = torch.split(imgs,[3,1],dim=0)
            vi_y,vi_cb,vi_cr = RGB2YCrCb(vi)
            outputs = torch.cat([vi_y, ir], dim=0) 
            return outputs, img_id
        else:
            vi = self.transforms(vi)  
            ir = self.transforms(ir)  
            vi_y,vi_cb,vi_cr = RGB2YCrCb(vi)
            output = torch.cat([vi_y, ir], dim=0)  
            return output, img_id,vi_cb,vi_cr

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.irs)