import cv2
import numpy as np
from dataset.L0_smooth import L0Smoothing

L0Smoothing_class = L0Smoothing(param_lambda=0.02)


def Laplacian_edge_detector(img):
    # input: [0, 1]
    # return: [0, 1] (H, W, 3)
    img = np.clip(img*255, 0, 255).astype(np.uint8) # (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Laplacian(img, cv2.CV_16S) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img

def Canny_edge_detector(img):
    # input: [0, 1]
    # return: [0, 1] (H, W, 3)
    img = np.clip(img*255, 0, 255).astype(np.uint8) # (H, W, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 50, 200) # (H, W)
    img = cv2.convertScaleAbs(img)
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, 2).repeat(3, axis=2) # (H, W, 3)
    return img

def L0_smooth(img):
    img = L0Smoothing_class.run(img)
    return img