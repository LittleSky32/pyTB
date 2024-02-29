import os
import random
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def trainData_load(img_dir: str, mask_dir: str, seed: int, dim: int, testSize: float):
    """
    load training data
    Args:
        img_dir: str for image file directory
        mask_dir: str for mask file directory
        seed: seed for reproduction
        dim: weight = height = dim
        testSize: ratio of test

    """
    # set data path
    image_path = os.path.join(img_dir)
    mask_path = os.path.join(mask_dir)

    # load image and mask
    images = os.listdir(img_dir)
    mask = os.listdir(mask_path)
    mask = [fName.split(".png")[0] for fName in mask]

    random.seed(seed)
    random.shuffle(mask)
    random.shuffle(images)

    # resize the images
    def getData(dim):
        """
        resize images
        Args:
            dim: height = weight = dim

        Returns:
        resized images and masks
        """
        im_array = []
        mask_array = []
        for i in tqdm(images):
            im_array.append(cv2.resize(cv2.imread(os.path.join(image_path, i)), (dim, dim))[:, :, 0])
            mask_array.append(cv2.resize(cv2.imread(os.path.join(mask_path, i)), (dim, dim))[:, :, 0])

        return im_array, mask_array

    # reshape the figures

    X_train, y_train = getData(dim)
    images = np.array(X_train).reshape(len(X_train), dim, dim, 1)
    mask = np.array(y_train).reshape(len(y_train), dim, dim, 1)

    # split the datasets into training set and validation set
    train_vol, test_vol, train_seg, test_seg = train_test_split((images - 127.0) / 127.0, # rescaling
                                                                (mask > 127).astype(np.float32),
                                                                test_size=testSize, random_state=seed)

    return train_vol, train_seg, test_vol, test_seg
