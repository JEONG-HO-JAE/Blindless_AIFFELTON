import os
import cv2
import numpy as np

def create_L_mask(image_path, g_channel, THRESHOLD):
    img = cv2.imread(image_path)
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)
    _, th_l = cv2.threshold(l, THRESHOLD, 255, cv2.THRESH_BINARY)
    return th_l

def applying_L_mask(image_path, g_channel, THRESHOLD):
    th_l = create_L_mask(image_path, g_channel, THRESHOLD)
    return cv2.bitwise_and(g_channel, th_l)

def crop_image(image):
    # Threshold the image to create a binary mask
    _, thresholded = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, find the largest contour (assumed to be the eye)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the region containing the eye
        cropped_eye = image[y:y + h, x:x + w]

        return cropped_eye
    else:
        return None