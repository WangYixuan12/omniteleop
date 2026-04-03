import cv2
import numpy as np


def concat_img_h(img_ls: list[np.ndarray]) -> np.ndarray:
    """Concatenate images horizontally.

    Args:
        img_ls (list[np.ndarray]): List of images to concatenate.

    Returns:
        np.ndarray: Concatenated image.
    """
    # Get the maximum height
    max_h = max(img.shape[0] for img in img_ls)

    # Resize images to the maximum height
    img_ls = [cv2.resize(img, (int(img.shape[1] * max_h / img.shape[0]), max_h)) for img in img_ls]

    # Concatenate images
    return cv2.hconcat(img_ls)


def concat_img_v(img_ls: list[np.ndarray]) -> np.ndarray:
    """Concatenate images vertically.

    Args:
        img_ls (list[np.ndarray]): List of images to concatenate.

    Returns:
        np.ndarray: Concatenated image.
    """
    # Get the maximum width
    max_w = max(img.shape[1] for img in img_ls)

    # Resize images to the maximum width
    img_ls = [cv2.resize(img, (max_w, int(img.shape[0] * max_w / img.shape[1]))) for img in img_ls]

    # Concatenate images
    return cv2.vconcat(img_ls)
