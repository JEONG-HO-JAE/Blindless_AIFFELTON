import cv2
import os
import numpy as np
import tensorflow as tf
from albumentations import HorizontalFlip, VerticalFlip, Compose, Resize

CLIPLIMIT = 3.0
GRIDSIZE = (8, 8)

def apply_clahe(g):
    clahe = cv2.createCLAHE(CLIPLIMIT, tileGridSize=GRIDSIZE)
    clahe_g_channel = clahe.apply(g)
    return clahe_g_channel

def red2green(r,g,b):
  tmp = r.copy()
  r = g.copy()
  g = tmp
  return r,g,b

def apply_normalization(g):
    min_val = np.min(g)
    max_val = np.max(g)
    normalized_g = (g - min_val) / (max_val - min_val)
    return normalized_g

def apply_cutomized_preprocess(image):
    r, g, b = cv2.split(image)
    bound = 100
    bound_pixel_count = (r >= bound).sum()
    if bound_pixel_count >= 0.5 * (r.shape[0] * r.shape[1]):
        rc,gc,bc = red2green(r,g,b)
        output_channel = cv2.addWeighted(g, 0.4, gc, 0.6, 0)
    else:
        output_channel = g
    
    output_channel = apply_clahe(output_channel)
    output_channel = apply_normalization(output_channel)
    return output_channel

def crop_black_part(image_path, image, label):
    file_name = os.path.basename(image_path)
    if "AFIO" in file_name:
        crop_img = image[10:1450, 150:1280]
        crop_label = label[10:1450, 150:1280]
    elif "DR_HAGIS" in file_name:
        if image.shape[0]==1944: # (1944, 2896, 3)
            crop_img = image[0:2100, 500:2380]
            crop_label = label[0:2100, 500:2380]
        elif image.shape[0]==2136: # (2136, 3216, 3)
            crop_img = image[0:2136, 450:2650]
            crop_label = label[0:2136, 450:2650]
        elif image.shape[0]==2304: # (2304, 3456, 3)
            crop_img = image[0:2304, 200:3256]
            crop_label = label[0:2304, 200:3256]
        elif image.shape[0]==1880: # (1880, 2816, 3)
            crop_img = image[0:1880, 200:2530]
            crop_label = label[0:1880, 200:2530]
        else: # (3168, 4752, 3)
            crop_img = image[0:3168, 800:3900]
            crop_label = label[0:3168,800:3900]
    elif "LES" in file_name:
        if image.shape[0]==1444: # (1444, 1620, 3)
            crop_img = image[0:1444, 60:1530]
            crop_label = label[0:1444, 60:1530]
    elif "STARE" in file_name:
        if image.shape[0]==605: # (605, 700, 3)
            crop_img = image[0:605, 20:680]
            crop_label = label[0:605, 20:680]
    elif "TREND" in file_name:
        if image.shape[0]==1920:
            crop_img = image[0:1920, 150:2400]
            crop_label = label[0:1920, 150:2400]
    else: 
        crop_img = image
        crop_label = label       
    return crop_img, crop_label

def build_augmentation_for_general(width=512, height=512, is_train=True):
    if is_train:
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Resize(width=width, height=height),
            
        ],is_check_shapes=False)
    else:
        return Compose([
            Resize(width=width, height=height),
        ],is_check_shapes=False)
        
def build_augmentation_for_adaptive(is_train=True):
    if is_train:
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            
        ],is_check_shapes=False)
    else:
        return Compose([
        ],is_check_shapes=False)

def resize_and_extract_patches(image, label, size, target_size):
    image = tf.expand_dims(image, axis=-1)
    label = tf.expand_dims(label, axis=-1)
    
    # 이미지 크기에 따라 조절
    resized_image = tf.image.resize(image, (target_size, target_size))
    resized_label = tf.image.resize(label, (target_size, target_size))
    # print(resized_image.shape)
    # print(resized_label.shape)
    # 이미지를 여러 하위 이미지로 자르기
    image_patches = tf.image.extract_patches(
        images=tf.expand_dims(resized_image, 0),
        sizes=[1, size, size, 1],
        strides=[1, size, size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    
    label_patches = tf.image.extract_patches(
        images=tf.expand_dims(resized_label, 0),
        sizes=[1, size, size, 1],
        strides=[1, size, size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    # print(image_patches.shape)
    # print(label_patches.shape)
    # 4D 텐서를 3D로 변환
    image_patches = tf.reshape(image_patches, (-1, size, size, 1))
    label_patches = tf.reshape(label_patches, (-1, size, size, 1))
    # print(image_patches.shape)
    # print(label_patches.shape)
    return image_patches, label_patches

def extract_patches(image, label, size):
    image = tf.expand_dims(image, axis=-1)
    label = tf.expand_dims(label, axis=-1)
    
    image_patches = tf.image.extract_patches(
        images=tf.expand_dims(image, 0),
        sizes=[1, size, size, 1],
        strides=[1, size, size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    
    label_patches = tf.image.extract_patches(
        images=tf.expand_dims(label, 0),
        sizes=[1, size, size, 1],
        strides=[1, size, size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    # print(image_patches.shape)
    # print(label_patches.shape)
    # 4D 텐서를 3D로 변환
    image_patches = tf.reshape(image_patches, (-1, size, size, 1))
    label_patches = tf.reshape(label_patches, (-1, size, size, 1))
    # print(image_patches.shape)
    # print(label_patches.shape)
    return image_patches, label_patches