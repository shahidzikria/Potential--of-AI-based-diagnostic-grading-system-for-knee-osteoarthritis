import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import zoom

INITIAL_SIZE = 224
IMG_SIZE = 224

def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (INITIAL_SIZE, INITIAL_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def fix_pixels(img):
    for i in range(INITIAL_SIZE):
        row = img[i, :]
        if len(set(row)) == 1:
            img[i] = np.zeros_like(row)
    return img

def sharpen_edges(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.addWeighted(img, 1.5, laplacian, -0.5, 0, dtype=cv2.CV_8U)
    return sharpened

def detect_knee_points(img, threshold=10):
    points, i = [], 0
    while i < INITIAL_SIZE:
        sum_i = int(np.sum(img[i]) / 255)
        if sum_i > threshold:
            points.append(i)
            j = i
            while j < INITIAL_SIZE:
                sum_j = int(np.sum(img[j]) / 255)
                if sum_j < threshold:
                    points.append(j)
                    break
                j += 1
            if len(points) % 2 != 0:
                points.append(j - 1)
            i = j + 1
        else:
            i += 1
    return points

def clean_points(points, threshold=30):
    sp = []
    if len(points) > 4:
        for i in range(0, len(points), 2):
            if (points[i+1] - points[i]) > threshold:
                sp.extend([points[i], points[i+1]])
    else:
        sp = points
    return sp

def one_hot_encode(labels, num_classes=5):
    num_samples = labels.shape[0]
    one_hot = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        one_hot[i, labels[i]] = 1
    return one_hot

def normalize_image(image, type='rescale'):
    if type == 'tf':
        img = image / 127.5 - 1.0
        return np.array(img)
    elif type == "rescale":
        return np.array(image / 255.0).astype(np.float32)
    else:
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std if std != 0 else 1)

# ---- Data Augmentation ----

def rot_img(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    if zoom_factor < 1:
        zh, zw = int(np.round(h * zoom_factor)), int(np.round(w * zoom_factor))
        top, left = (h - zh) // 2, (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
    elif zoom_factor > 1:
        zh, zw = int(np.round(h / zoom_factor)), int(np.round(w / zoom_factor))
        top, left = (h - zh) // 2, (w - zw) // 2
        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        trim_top = (out.shape[0] - h) // 2
        trim_left = (out.shape[1] - w) // 2
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = img
    return out

def augmentation(imgs, lbls, rotate_ang=[45, 315], zooming=[0.9, 1.1]):
    new_X, new_y = [], []
    def process_image(i):
        augmented_images = [imgs[i], cv2.flip(imgs[i], 1)]
        augmented_labels = [lbls[i], lbls[i]]
        for ang in rotate_ang:
            augmented_images.append(rot_img(imgs[i], ang))
            augmented_labels.append(lbls[i])
        for zm in zooming:
            augmented_images.append(clipped_zoom(imgs[i], zm))
            augmented_labels.append(lbls[i])
        return augmented_images, augmented_labels

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_image, range(len(imgs)))

    for res in results:
        new_X.extend(res[0])
        new_y.extend(res[1])
    return np.array(new_X), np.array(new_y)
