import cv2
import math
import numpy as np

from albumentations import  HorizontalFlip, Compose, Resize

CLIPLIMIT = 3.0
GRIDSIZE = (8, 8)
WIDTH=256
HEIGHT=256

def apply_gamma_correction(img, mid):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mean = np.mean(val)
    gamma = math.log(mid * 255) / math.log(mean)

    val_gamma = np.power(val, gamma).clip(0, 255).astype(np.uint8)
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    img_gamma = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
    return img_gamma 

def apply_clahe (img, Limit, GridSize):
  b, g, r = cv2.split(img)
  clahe = cv2.createCLAHE(clipLimit=Limit, tileGridSize=GridSize)
  clahe_g_channel = clahe.apply(g)
  clahe_g_channel = clahe_g_channel.reshape((clahe_g_channel.shape[0], clahe_g_channel.shape[1], 1))
  return clahe_g_channel 

def apply_normalization(img):
    min_val = np.min(img)
    max_val = np.max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    normalized_img = normalized_img.reshape((normalized_img.shape[0], normalized_img.shape[1], 1))
    return normalized_img

def build_augmentation(is_train=True):
    if is_train:
        return Compose([
            HorizontalFlip(p=0.5),
            Resize(width=WIDTH, height=HEIGHT),
            
        ],is_check_shapes=False)
    else:
        return Compose([
            Resize(width=WIDTH, height=HEIGHT),
        ],is_check_shapes=False)

def apply_preprocessing(augmented):
    # Apply CLAHE (replace with your implementation)
    augmented['image'] = apply_clahe(augmented['image'], CLIPLIMIT, GRIDSIZE)

    # Apply normalization
    augmented['image'] = apply_normalization(augmented['image'])