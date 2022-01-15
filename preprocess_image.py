import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def find_y(
    img: np.ndarray, thres: float, tol: float = 10, gthres: float = 0.1
) -> tuple:
    """Find minimum top and bottom points to crop using threshold of pixel values by looping rows of image.

    Args:
        img (np.ndarray): Image to be cropped.
        thres (float): Threshold of pixel value.
        tol (float, optional): Tolerance. Number of rows to be torelance from the minimum points. Defaults to 10.
        gthres (float, optional): Factor to multiple with number of columns to get threshold of number of valid pixels. Defaults to 0.1.

    Returns:
        tuple: Minimum top and bottom points.
    """

    y1, y2 = 0, 0

    # y1 is minimum top
    for row in range(img.shape[0]):
        if img[row].max() > thres:
            # 2d array that counts valid pixels that exceed threshold, [[  0 M], [  1  N]]
            gtruth = np.array(np.unique((img[row] > thres), return_counts=True)).T
            # number of valid pixels must exceed at least gthres of total columns
            if gtruth[-1][-1] > gthres * img.shape[1]:
                if row - tol < 0:
                    y1 = 0
                else:
                    y1 = row - tol
                break

    # y2 is minimum bottom
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


def find_x(img: np.ndarray, thres: float) -> tuple:
    """Find minimum left and right points to crop using threshold of pixel values by looping columns of image.

    Args:
        img (np.ndarray): Image to be cropped.
        thres (float): Threshold of pixel value.

    Returns:
        tuple: Minimum left and right points.
    """

    x1, x2 = 0, 0

    # x1 is minimum left
    for col in range(img.shape[1]):
        if img[:, col].max() > thres:
            x1 = col
            break

    # x2 is minimum right
    for col in sorted(range(img.shape[1]), reverse=True):
        if img[:, col].max() > thres:
            x2 = col
            break

    return x1, x2


def preprocess(image: str, save_path: str):
    """preprocess data by cropping out the waveform.

    Args:
        image (str): image path.
        save_path (str): path to save preprocessed image.
    """

    filename = image.split("\\")[-1]

    with open(image, "rb") as f:
        img = Image.open(f)
        img = img.convert("L")  # convert to grayscale image

    # crop waveform draftly
    w, h = img.size  # w is width, h is height
    x1 = 0.1 * w  # left crop from left
    x2 = 0.85 * w  # right crop from left
    y1 = 0.51 * h  # top crop from top
    y2 = 0.94 * h  # bottom crop from top
    img = img.crop((x1, y1, x2, y2))

    # crop waveform precisely
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img_mat = np.array(img)
    x1, x2 = find_x(img_mat, thres=100)
    y1, y2 = find_y(img_mat, thres=60, gthres=0.075)
    img = img.crop((x1, y1, x2, y2))
    c = 1
    img = img.resize([i * c for i in [448, 112]])

    # save image
    img.save(save_path + filename)


if __name__ == "__main__":
    RAW_PATH = "data/raw/dataset/"
    PREPROCESS_PATH = "data/preprocess/"

    if not os.path.exists(PREPROCESS_PATH):
        os.makedirs(PREPROCESS_PATH)

    data_dir = Path(RAW_PATH)
    data = [str(image) for image in list(data_dir.glob("*/*.jpg"))]

    for image in data:
        preprocess(image, PREPROCESS_PATH)

    print("finish")
