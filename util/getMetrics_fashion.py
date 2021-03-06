import os
from inception_score import get_inception_score

from skimage.io import imread, imsave
from skimage.measure import compare_ssim

import numpy as np
import pandas as pd

from tqdm import tqdm
import re
import cv2

from PIL import Image
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir",
                        type=str,
                        default="/home/lchen63/code/pix2pixhd_l/results/deepfashion_man/test_latest/images/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')


    return parser.parse_args()

config = parse_args()

def l1_score(generated_images, reference_images):
    score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        score = np.abs(2 * (reference_image/255.0 - 0.5) - 2 * (generated_image/255.0 - 0.5)).mean()
        score_list.append(score)
    return np.mean(score_list)


def ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in zip(reference_images, generated_images):
        ssim = compare_ssim(reference_image, generated_image, gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    return np.mean(ssim_score_list)


def save_images(input_images, target_images, generated_images, names, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for images in zip(input_images, target_images, generated_images, names):
        res_name = str('_'.join(images[-1])) + '.png'
        imsave(os.path.join(output_folder, res_name), np.concatenate(images[:-1], axis=1))


def create_masked_image(names, images, annotation_file):
    import pose_utils
    masked_images = []
    df = pd.read_csv(annotation_file, sep=':')
    for name, image in zip(names, images):
        to = name[1]
        ano_to = df[df['name'] == to].iloc[0]

        kp_to = pose_utils.load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = pose_utils.produce_ma_mask(kp_to, image.shape[:2])
        masked_images.append(image * mask[..., np.newaxis])

    return masked_images



def addBounding(image, bound=40):
    h, w, c = image.shape
    image_bound = np.ones((h, w+bound*2, c))*255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound+w] = image

    return image_bound



def load_generated_images(images_folder):
    input_images = []
    target_images = []
    generated_images = []

    names = []

    for gg in os.listdir(images_folder):
        if '_synthesized_image.jpg' in gg:
            # print (gg[:-22] + '_real_image.jpg')
            # if os.path.exists(gg[:-22] + '_real_image.jpg'):
                # generated_images.append(gg)
                # target_images.append(gg[:-22] + '_real_image.jpg')
                # input_images.append(gg[:-22] + '_input_image.jpg')

            names.append(gg[:-22])
    for img_name in names:
        fake_img = cv2.imread(os.path.join(images_folder, img_name + "_synthesized_image.jpg" ))

        input_images.append(cv2.imread(os.path.join(images_folder, img_name + '_input_image.jpg' )))
        target_images.append(cv2.imread(os.path.join(images_folder, img_name + '_real_image.jpg' )))

        generated_images.append(fake_img)
    # print (type(generated_images[0]))

    return input_images, target_images, generated_images


def test(generated_images_dir):
    # load images
    print ("Loading images...")

    input_images, target_images, generated_images = load_generated_images(generated_images_dir)

    print ("Compute inception score...")
    inception_score = get_inception_score(generated_images)
    print ("Inception score %s" % inception_score[0])

    print ("Compute structured similarity score (SSIM)...")
    structured_score = ssim_score(generated_images, target_images)
    print ("SSIM score %s" % structured_score)

    print ("Inception score = %s; SSIM score = %s" % (inception_score, structured_score))


if __name__ == "__main__":
    # fix these paths
    generated_images_dir = config.sample_dir

    test(generated_images_dir)




