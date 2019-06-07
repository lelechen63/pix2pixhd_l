### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
# from util.visualizer import Visualizer
#from util import html
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
from networks import deeplab_xception_transfer, graph
import custom_transforms as tr
#from vis import *
import timeit
import torch.nn.functional as F



def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample



def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):

    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


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

    vis_res = decode_labels(new)
    parsing_im = Image.fromarray(vis_res[0])
    parsing_im.save(parsing_path.replace('1.png', '2.png'))
    return Image.fromarray(new[0])



def image_resizer(image_path):
    img  = Image.open(image_path)
    size = (max(img.size),)*2
    layer = Image.new('RGB', size, (255,255,255))

    layer.paste(img, tuple(map(lambda x:(x[0]-x[1])//2, zip(size, img.size))))
    layer = layer.resize((512,512))
    layer.save(image_path.replace('.jpg', '_512.jpg'))

    # return layer


def garment_parsing(img_path):

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    x = torch.load('./checkpoints/universal_trained.pth')
    net.load_source_model(x)
    net.cuda()

    img_path_splits = img_path.split('/')
    output_path = os.path.join('/', *img_path_splits[:-1])

    output_name = img_path_splits[-1].split('.')[0]
    use_gpu = True


    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img  = Image.open(img_path).convert('RGB') 
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if use_gpu >= 0:
                inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
               outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    shit = {}
    for i in range(512):
        for j in range(512):
            if results[0,i,j] not in shit.keys():
                shit[results[0,i,j]]= 1
            else:
                shit[results[0,i,j]] += 1
    vis_res = decode_labels(results)    
    parsing_im = Image.fromarray(vis_res[0])
  
    cv2.imwrite(output_path+'/{}_parsing1.png'.format(output_name), results[0])
    parsing_im.save(output_path+'/{}_parsing.png'.format(output_name))
    end_time = timeit.default_timer()

def demo(img_path):
    opt = TestOptions().parse(save=False)


    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    
   
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)


    # input 1 : front image
    '''resize input image, new name with  _512 '''
    image_resizer(img_path)

    input_image = img_path[:-4] + '_512.jpg'

    # get garment parsing, save it save _parsing1.png (gray scale), _parsing.png
    garment_parsing(input_image)



    # input 2 (garment parsing)
    garment =  input_image.replace('.jpg', '_parsing1.png')

    # generate back parsing
    gt_garment = front2back_mapping(input_image.replace('.jpg', '_parsing1.png'))


    A = Image.open(garment)        
    params = get_params(opt, A.size)
    segment = util.PIL2array(A).copy()
    segment[segment>0] = 1
    if opt.label_nc == 0:
        transform_A = get_transform(opt, params)
        A = A.convert('RGB')
        in_tensor = transform_A(A)
        
    else:
        transform_A = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        in_tensor = transform_A(A) * 255.0

    B = Image.open(input_image).convert('RGB')    
    transform_B = get_transform(opt, params)  
    B = B * segment
    B = Image.fromarray(B)    
    in_img_tensor = transform_B(B)



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

    

    data = {'input_parsing': in_tensor.unsqueeze(0),  'input_image': in_img_tensor.unsqueeze(0),
        'gt_parsing': gt_tensor.unsqueeze(0)}

    if opt.data_type == 16:
        data['input_parsing'] = data['input_parsing'].half()
        data['gt_parsing']  = data['gt_parsing'].half()
    elif opt.data_type == 8:
        data['input_parsing'] = data['input_parsing'].uint8()
        data['gt_parsing']  = data['gt_parsing'].uint8()


    minibatch = 1     
    generated = model.inference(data['input_parsing'], data['input_image'], data['gt_parsing'], data['input_image'])


    fake_path =  input_image.replace('.jpg', '_back.png')
    fake = util.tensor2im(generated.data[0])
    util.save_image(fake, fake_path)

            

    return  input_image, fake_path, input_image.replace('.jpg', '_parsing.png'),  input_image.replace('.jpg', '_parsing2.png')

front_img = ['/home/lchen63/data_test/lele_f.jpg',
    # '/home/lchen63/data_test/Zhong_wg.jpg',
    '/home/lchen63/data_test/Shuang1.jpg',
    '/home/lchen63/data_test/yuxin_wg.jpg',
    # '/home/lchen63/data_test/fashion2.png',
    # '/home/lchen63/data_test/01_1_front.jpg',
    '/home/lchen63/data_test/04_1_front.jpg',
    '/home/lchen63/data_test/amazon_fashion1.jpg']

  
for i in front_img:
    demo(i)
