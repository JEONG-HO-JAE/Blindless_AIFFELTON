import cv2
import numpy as np
import tensorflow as tf
from albumentations import HorizontalFlip, VerticalFlip, Compose, Resize, RandomCrop

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