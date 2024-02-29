import os
import random
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

sys.path.append('.')
import segmentation_model
from tqdm import tqdm
import matplotlib.pyplot as plt


def testData_load(pred_dir: str, dim: int):
    """
    load prediction file
    Args:
        pred_dir: the file including images to be predicted
        dim: height = width = dim

    Returns:
        pred_images: original images
        pred_vol: processed images
    """
    # set data path
    pred_path = os.path.join(pred_dir)
    pred_images = os.listdir(pred_dir)

    # load data
    def getTestData(dim):
        pred_array = []
        for i in tqdm(pred_images):
            pred_array.append(cv2.resize(cv2.imread(os.path.join(pred_path, i)), (dim, dim))[:, :, 0])

        return pred_array

    # process the images
    pred = getTestData(dim)
    pred_vol = np.array(pred).reshape(len(pred), dim, dim, 1)
    pred_vol = (pred_vol - 127.0) / 127.0

    return pred_images, pred_vol


def segmentation_pred(pred_dir, model_path, pred_output_path, dim=256):
    """
    predict segmentation, and masked image will be saved
    Args:
        pred_dir: path where prediction file is
        model_path: path where model is saved
        pred_output_path: path to save prediction output
        dim: height = width = dim, default is 256

    """
    pred_images, pred_vol = testData_load(pred_dir, dim)
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_coef_loss': segmentation_model.dice_coef_loss})
    preds = model.predict(pred_vol)
    print('Prediction start!')
    for i in tqdm(range(len(preds))):
        plt.imshow(np.squeeze(preds[i]))
        plt.title(pred_images[i] + ' Prediction')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(pred_output_path + 'mask_' + pred_images[i])
    print('Predicted masked figures are saved at ' + pred_output_path)
    return
