### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image
import pickle
import random
import torchvision.transforms as transforms

class DenseposeDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.train = opt.train
        _file = open(os.path.join(self.root, "pickle/man_tee_train.pkl"), "rb")
        self.training = pickle.load(_file)
        self.train_data = []
        for key in self.training.keys():
            self.train_data.append(self.training[key])
            # for img in self.train[key]:
            #     self.train_data.append(img.replace('.jpg','_512.jpg'))
        _file.close()

        random.shuffle(self.train_data)
        _file = open(os.path.join(self.root, "pickle/man_tee_test.pkl"), "rb")
        self.test = pickle.load(_file)
        self.test_data = []
        for key in self.test.keys():
            self.test_data.append(self.test[key])
            # for img in self.test[key]:
                # self.test_data.append(img.replace('.jpg','_512.jpg'))
        _file.close()
        random.shuffle(self.test_data)
        # self.dataset_size = len(self.A_paths) 
        if self.train == 'train':
            self.dataset_size  = len(self.train_data)
        else:
            self.dataset_size  = len(self.test_data)
    
    def __getitem__(self, index):                   
        in_img_tensor = inst_tensor = feat_tensor = 0
     
        ### input A (label maps)

        # input 1 : front image
        if self.train == 'train':
            self.input_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.train_data[index][0].replace('.jpg','_512.jpg')) 
        else:
            self.input_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.test_data[index][0].replace('.jpg','_512.jpg')) 

        

        # input 2 (garment parsing)
        self.garment =  self.input_image.replace('.jpg', '_parsing1.png')
        A = Image.open(self.garment)        
        # A =  A.transpose(Image.FLIP_LEFT_RIGHT)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A = A.convert('RGB')
            A = transforms.functional.affine(A, params['angle'], params['translate'], params['scale'], params['shear'] )
            in_tensor = transform_A(A)
            
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A = transforms.functional.affine(A, params['angle'], params['translate'], params['scale'], params['shear'] )
            in_tensor = transform_A(A) * 255.0

        B = Image.open(self.input_image).convert('RGB')
        # params = get_params(self.opt, B.size)
        transform_B = get_transform(self.opt, params)  
        B = transforms.functional.affine(B, params['angle'], params['translate'], params['scale'], params['shear'] )    
        in_img_tensor = transform_B(B)

        # gt view can be back size view or side view
        gt_view = random.choice([x for x in range(1,3)])

        if self.train =='train':
            self.gt_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.train_data[index][gt_view].replace('.jpg','_512.jpg')) 
        else:
            self.gt_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.test_data[index][gt_view].replace('.jpg','_512.jpg')) 

        C = Image.open(self.gt_image).convert('RGB')
        # params = get_params(self.opt ,C.size)

        transform_C = get_transform(self.opt, params)
        C = transforms.functional.affine(C, params['angle'], params['translate'], params['scale'], params['shear'] )       
        out_img_tensor = transform_C(C)


        # gt garment parsing
        self.gt_garment =  self.gt_image.replace('.jpg', '_parsing1.png')
        gt_garment = Image.open(self.gt_garment)

        # gt_garment =  gt_garment.transpose(Image.FLIP_LEFT_RIGHT)        
        # params = get_params(self.opt, gt_garment.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            gt_garment =  gt_garment.convert('RGB')
            gt_garment =  transforms.functional.affine(gt_garment, params['angle'], params['translate'], params['scale'], params['shear'] ) 
            A_tensor = transform_A(gt_garment)
        else:
            transform_D = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
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
        return 'DenseposeDataset'