### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import argparse
from PIL import Image
import cv2
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image
import pickle
import random
import torchvision.transforms as transforms
import numpy as np
opt = TestOptions().parse(save=False)


opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx


back_img = '/home/lchen63/data_test/lele_b.jpg'
front_img = ['/home/lchen63/data_test/lele_f.jpg',
'/home/lchen63/data_test/Zhong_wg.jpg',
'/home/lchen63/data_test/Shuang1.jpg',
'/home/lchen63/data_test/yuxin_wg.jpg',
'/home/lchen63/data_test/fashion2.png',
'/home/lchen63/data_test/01_1_front.jpg',
'/home/lchen63/data_test/04_1_front.jpg',
'/home/lchen63/data_test/amazon_fashion1.jpg']

def front2back_mapping(parsing_path):
    img_parsing  = cv2.imread(parsing_path, 0)

    new1 = img_parsing.copy()

    new  = np.zeros(( 1, 512, 512), dtype=np.uint8)
    new[0,:,:] = new1
    high_neck = False
    
    for i in range(512):
        high_cloth = False
        hair = False
        for j in range(512): 
            if img_parsing[i,j] == 5:
                high_cloth = True
            elif  img_parsing[i,j] == 10:
                high_neck = True
            elif img_parsing[i,j] == 2:
                hair = True
            if img_parsing[i,j] == 15:
                new[0,i,j] = 14
            elif img_parsing[i,j] == 14:
                new[0,i,j] = 15
            elif img_parsing[i,j] == 18:
                new[0,i,j] = 19
            elif img_parsing[i,j] == 19:
                new[0,i,j] = 18
            elif  img_parsing[i,j] == 10:

                if  high_cloth:
                    new[0,i,j] = 5
                else:
                    new[0,i,j] = 10
                if hair:
                    new[0,i,j] = 2
            elif img_parsing[i,j] == 13:
                if high_neck:

                    new[0,i,j] = 10
                else:
                    new[0,i,j] = 2
                if hair:
                    new[0,i,j] = 2
            elif img_parsing[i,j] == 5:
                new[0,i,j] = 5
            elif img_parsing[i,j] == 9:
                new[0,i,j] = 9
            elif img_parsing[i,j] == 2:
                new[0,i,j] = 2


    return Image.fromarray(new[0])
    # tmp  = np.zeros((1, h, w), dtype=np.uint8)
    # tmp[0,:,: ] = rotated_img
    # vis_res = decode_labels(new)
    # parsing_im = Image.fromarray(vis_res[0])
    # parsing_im.save(parsing_path.replace('1.png', '2.png'))

front2back_mapping('/home/lchen63/data_test/Shuang1_512_parsing1.png')

for jj in front_img:
    opt.identity_image = jj

    opt.pose_image = back_img


    in_img_tensor = inst_tensor = feat_tensor = 0     
    ### input A (label maps)
    # input 1 : front image
    input_image = opt.identity_image[:-4] + '_512.jpg'


    # input 2 (garment parsing)
    garment =  input_image.replace('.jpg', '_parsing1.png')

    gt_garment = front2back_mapping(input_image.replace('.jpg', '_parsing1.png'))


    A = Image.open(garment)        
    params = get_params(opt, A.size)
    segment = util.PIL2array(A).copy()
    segment[segment>0] = 1
    if opt.label_nc == 0:
        transform_A = get_transform(opt, params)
        A = A.convert('RGB')
        # A = transforms.functional.affine(A, params['angle'], params['translate'], params['scale'], params['shear'] )
        in_tensor = transform_A(A)
        
    else:
        transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        # A = transforms.functional.affine(A, params['angle'], params['translate'], params['scale'], params['shear'] )
        in_tensor = transform_A(A) * 255.0

    B = Image.open(input_image).convert('RGB')
    # params = get_params(self.opt, B.size)
    transform_B = get_transform(opt, params)  
    B = B * segment
    B = Image.fromarray(B)    
    in_img_tensor = transform_B(B)

#gt image
    gt_image = opt.pose_image[:-4] + '_512.jpg'

# gt garment parsing
    gt_segment = util.PIL2array(gt_garment).copy()
    gt_segment[gt_segment>0] = 1
    if opt.label_nc == 0:
        transform_A = get_transform(opt, params)
        gt_garment =  gt_garment.convert('RGB')
        A_tensor = transform_A(gt_garment)
    else:
        transform_D = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        gt_tensor = transform_D(gt_garment) * 255.0

    C = Image.open(gt_image)
    C = C * gt_segment
    C = Image.fromarray(C).convert('RGB')
    transform_C = get_transform(opt, params)
    out_img_tensor = transform_C(C)

    data = {'input_parsing': in_tensor.unsqueeze(0),  'input_image': in_img_tensor.unsqueeze(0),
        'gt_parsing': gt_tensor.unsqueeze(0), 'gt_image': out_img_tensor.unsqueeze(0), 'input_path': input_image, 'gt_path':gt_image}

    if opt.data_type == 16:
        data['input_parsing'] = data['input_parsing'].half()
        data['gt_parsing']  = data['gt_parsing'].half()
    elif opt.data_type == 8:
        data['input_parsing'] = data['input_parsing'].uint8()
        data['gt_parsing']  = data['gt_parsing'].uint8()




    minibatch = 1     
    generated = model.inference(data['input_parsing'], data['input_image'], data['gt_parsing'],data['gt_image'])
        

    # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           # ('synthesized_image', util.tensor2im(generated.data[0]))])
    visuals = OrderedDict([('input_image', util.tensor2im(data['input_image'][0])),
                                    ('input_parsing', util.tensor2label(data['input_parsing'][0],  opt.label_nc)),
                                    ('gt_label', util.tensor2label(data['gt_parsing'][0], opt.label_nc)),
                                    ('synthesized_image', util.tensor2im(generated.data[0])),
                                    ('real_image', util.tensor2im(data['gt_image'][0]))])
    img_path = data['input_path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, [img_path])

webpage.save()
