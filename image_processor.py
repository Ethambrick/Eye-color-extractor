import cv2
import numpy as np
import os


def load_image(image_path):
    img = cv2.imread(image_path)
    return img


def get_eye_crop(image, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = int(min(xs)), int(max(xs))
    min_y, max_y = int(min(ys)), int(max(ys))

    return image[min_y:max_y, min_x:max_x]

def extract_color_features(eye_img):
    eye_img = cv2.resize(eye_img, (50, 50))
    pixels = eye_img.reshape(-1, 3)

    # Split channels
    blues = pixels[:, 0]
    greens = pixels[:, 1]
    reds = pixels[:, 2]

    features = [
        np.mean(reds),
        np.mean(greens),
        np.mean(blues),
        np.std(reds),
        np.std(greens),
        np.std(blues)
    ]

    return features

def extract_color_distribution(eye_img):
    import numpy as np

    eye_img = cv2.resize(eye_img, (50, 50))
    pixels = eye_img.reshape(-1, 3)

    brown = 0
    blue = 0
    green = 0

    for (b, g, r) in pixels:
        if r > g and r > b:
            brown += 1
        elif b > r and b > g:
            blue += 1
        elif g > r and g > b:
            green += 1

    total = len(pixels)

    return [
        brown / total,
        blue / total,
        green / total
    ]