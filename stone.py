import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import math

s = r'images/final_stone1.jpg'
img = cv2.imread(s, 0)

# Check if image is loaded correctly
if img is None:
    raise FileNotFoundError(f"Image not found at path: {s}")

def histeq(img):
    histogram = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram[int(img[i, j])] += 1
    cum = np.cumsum(histogram)
    cum = (cum - cum.min()) * 255 / (cum.max() - cum.min())
    new_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i, j] = cum[int(img[i, j])]
    return new_img

def laplacian_filter(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    result = cv2.filter2D(img, -1, kernel)
    return np.clip(result, 0, 255)

def gaussian_filter(img):
    result = cv2.GaussianBlur(img, (5, 5), 0)
    return result

def median_filter(img):
    result = cv2.medianBlur(img, 5)
    return result

def segmentation(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    return erosion, dilation

def pre_processing(img):
    img_rm_small = morphology.remove_small_objects(img > 0, min_size=5, connectivity=2)
    thresh = cv2.threshold(img_rm_small.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    filled = morphology.remove_small_holes(thresh.astype(bool), area_threshold=100)
    return (filled.astype(np.uint8) * 255)

def enhancement(img):
    img_gaussian = gaussian_filter(img)
    img_median = median_filter(img_gaussian)
    return img_median

def detection(img):
    preprocessed = pre_processing(img)
    enhanced = enhancement(preprocessed)
    erosion, dilation = segmentation(enhanced)
    plt.figure('Detection')
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Enhanced Image')
    plt.imshow(enhanced, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Segmentation')
    plt.imshow(dilation, cmap='gray')
    plt.show()

detection(img)
