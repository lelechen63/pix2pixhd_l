### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image

class DenseposeDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.train = opt.train
        _file = open(os.path.join(self.root, "pickle/man_tee_train.pkl"), "rb")
        self.train = pickle.load(_file)
        self.train_data = []
        for key in self.train.keys():
            self.train_data.append(self.train[key])
            # for img in self.train[key]:
            #     self.train_data.append(img.replace('.jpg','_512.jpg'))
        _file.close()


        _file = open(os.path.join(self.root, "pickle/man_tee_test.pkl"), "rb")
        self.test = pickle.load(_file)
        self.test_data = []
        for key in self.test.keys():
            self.test_data.append(self.test[key])
            # for img in self.test[key]:
                # self.test_data.append(img.replace('.jpg','_512.jpg'))
        _file.close()
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
            self.input_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.train_data[index][0]) 
        else:
            self.input_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.test_data[index][0]) 

        B = Image.open(self.input_image).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        in_img_tensor = transform_B(B)


        # input 2 (garment parsing)
        self.garment =  self.input_image.replace('.jpg', '_parsing3.png')
        A = Image.open(self.garment)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            in_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            in_tensor = transform_A(A) * 255.0

        current_view = int(self.train_data[index].split('_')[1])


        # gt view can be back size view or side view
        gt_view = random.choice(x for x in range(1,2))

        if self.train =='train':
            self.gt_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.train_data[index][gt_view]) 
        else:
            self.gt_image = os.path.join(self.root, 'MEN','Tees_Tanks',  self.test_data[index][gt_view]) 

        B = Image.open(self.gt_image).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        out_img_tensor = transform_B(B)


        # gt garment parsing
        self.gt_garment =  self.gt_image.replace('.jpg', '_parsing3.png')
        gt_garment = Image.open(self.gt_garment)        
        params = get_params(self.opt, gt_garment.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(gt_garment.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            gt_tensor = transform_A(gt_garment) * 255.0
        print (in_tensor.size())    
        print ('ggggggggggg')                   
        input_dict = {'input_parsing': in_tensor,  'input_image': in_img_tensor,
            'gt_parsing': gt_tensor, 'gt_image': out_img_tensor, 'input path': self.input_image, 'gt path': self.gt_image}

        return input_dict

    def __len__(self):
        if self.train == 'train':
            return len(self.train_data) // self.opt.batchSize * self.opt.batchSize
        else:
            return len(self.test_data) // self.opt.batchSize * self.opt.batchSize
    def name(self):
        return 'DenseposeDataset'