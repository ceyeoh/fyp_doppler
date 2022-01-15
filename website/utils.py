import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def find_y(img, thres, tol=10, gthres=0.1):
    y1, y2 = 0, 0

    for row in range(img.shape[0]):
        if img[row].max() > thres:
            gtruth = np.array(np.unique((img[row] > thres), return_counts=True)).T
            if gtruth[-1][-1] > gthres * img.shape[1]:
                if row - tol < 0:
                    y1 = 0
                else:
                    y1 = row - tol
                break

    for row in sorted(range(img.shape[0]), reverse=True):
        if img[row].max() > thres:
            gtruth = np.array(np.unique((img[row] > thres), return_counts=True)).T
            if gtruth[-1][-1] > gthres * img.shape[1]:
                if row + tol > img.shape[0]:
                    y2 = img.shape[0]
                else:
                    y2 = row + tol
                break

    return y1, y2


def find_x(img, thres):
    x1, x2 = 0, 0

    for col in range(img.shape[1]):
        if img[:, col].max() > thres:
            x1 = col
            break

    for col in sorted(range(img.shape[1]), reverse=True):
        if img[:, col].max() > thres:
            x2 = col
            break

    return x1, x2


def loader(filename):
    img = Image.open(filename)
    img = img.convert("L")
    w, h = img.size
    x1 = 0.1 * w
    x2 = 0.85 * w
    y1 = 0.51 * h
    y2 = 0.94 * h
    img = img.crop((x1, y1, x2, y2))
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img_mat = np.array(img)
    x1, x2 = find_x(img_mat, thres=100)
    y1, y2 = find_y(img_mat, thres=60, gthres=0.075)
    img = img.crop((x1, y1, x2, y2))
    c = 1
    img = img.resize([i * c for i in [448, 112]])
    return img.convert("RGB")
