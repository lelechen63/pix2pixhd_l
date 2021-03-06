### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image
import pickle
import random
import torchvision.transforms as transforms

class SynDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = '/data/syn'    
        self.train = opt.train
        _file = open(os.path.join(self.root, "pickle/train2.pkl"), "rb")
        self.train_data = pickle.load(_file)
       
        _file.close()

        random.shuffle(self.train_data)
        _file = open(os.path.join(self.root, "pickle/test12.pkl"), "rb")
        self.test_data = pickle.load(_file)
       
        _file.close()
        # random.shuffle(self.test_data)


        if self.train == 'train':
            self.dataset_size  = len(self.train_data)
        else:
            self.dataset_size  = len(self.test_data)
    
    def __getitem__(self, index): 
        in_img_tensor = inst_tensor = feat_tensor = 0
     
        ### input A (label maps)

        # input 1 : front image
        if self.train == 'train':
            self.input_image = os.path.join(self.root, 'data',  self.train_data[index], 'model_frontal.png') 
        else:
            self.input_image = os.path.join(self.root, 'data',  self.test_data[index], 'model_frontal.png') 

        # input 2 (garment parsing)
        self.garment =  self.input_image.replace('.png', '_parsing1.png')
        A = Image.open(self.garment)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        if self.train == 'train':
            A = transforms.functional.affine(A, params['angle'], params['translate'], params['scale'], params['shear'] )
        in_tensor = transform_A(A) * 255.0


        B = Image.open(self.input_image).convert('RGB')
        transform_B = get_transform(self.opt, params)  
        if self.train == 'train':
            B = transforms.functional.affine(B, params['angle'], params['translate'], params['scale'], params['shear'] )    
        in_img_tensor = transform_B(B)

        
        if self.train =='train':
            self.gt_image = os.path.join(self.root, 'data',  self.train_data[index], 'model_back.png') 
        else:
            self.gt_image = os.path.join(self.root, 'data',  self.test_data[index], 'model_back.png') 

        C = Image.open(self.gt_image).convert('RGB')
        # params = get_params(self.opt ,C.size)

        transform_C = get_transform(self.opt, params)

        if self.train == 'train':
            C = transforms.functional.affine(C, params['angle'], params['translate'], params['scale'], params['shear'] )       
        out_img_tensor = transform_C(C)

# gt garment parsing
        self.gt_garment =  self.gt_image.replace('.png', '_parsing1.png')
        gt_garment = Image.open(self.gt_garment)

        transform_D = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        if self.train =='train':
            gt_garment = transforms.functional.affine(gt_garment, params['angle'], params['translate'], params['scale'], params['shear'] ) 
        gt_tensor = transform_D(gt_garment) * 255.0
        input_dict = {'input_parsing': in_tensor,  'input_image': in_img_tensor,
            'gt_parsing': gt_tensor, 'gt_image': out_img_tensor, 'input_path': self.input_image, 'gt_path': self.gt_image}

        return input_dict

    def __len__(self):
        if self.train == 'train':
            return len(self.train_data) // self.opt.batchSize * self.opt.batchSize
        else:
            return len(self.test_data) // self.opt.batchSize * self.opt.batchSize
    def name(self):
        return 'SynDataset'