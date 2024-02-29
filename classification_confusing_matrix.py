import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../Documents/WeChat Files/wxid_kfnmvvjdq3hs22/FileStorage/File/2022-12')
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn.metrics import classification_report


def trainData_load(metadir1: str, metadir2: str, img_dir1: str, img_dir2: str,
                   seed=1, SHUFFLE=1000, BATCH=5, test_ratio=0.2, dim=192):
    """
    Load the training data
    Args:
        metadir1: metadata file for the normal images
        metadir2: metadata file for the TB images
        img_dir1: image directory for normal images
        img_dir2: image directory for TB images
        seed: seed is used to reproduce the same result
        SHUFFLE: seed specifical for shuffle
        BATCH: batch size
        test_ratio: ratio of data used for test
        dim: the input size of image (height, weight = dim)

    Returns:
        trainData, testData, vaildData stands for training, test, validation dataset
    """

    # Reproduce the loadiing of training data
    nor_data = pd.read_csv(metadir1)

    unor_data = pd.read_csv(metadir2)

    nor_dataPath = [img_dir1 + nor_path + '.png' for nor_path in nor_data.iloc[:, 0]]
    nor_dataLabel = [0] * len(nor_dataPath)

    unor_dataPath = [img_dir2 + unor_path + '.png' for unor_path in unor_data.iloc[:, 0]]
    unor_dataLabel = [1] * len(unor_dataPath)

    # equalize the datasets
    maxlen = min(len(nor_dataPath), len(unor_dataPath))
    dataPath = nor_dataPath[0:maxlen] + unor_dataPath[0:maxlen]
    dataLabel = nor_dataLabel[0:maxlen] + unor_dataLabel[0:maxlen]

    random.seed(seed)
    random.shuffle(dataPath)
    random.shuffle(dataLabel)
    dataPath = dataPath
    dataLabel = dataLabel

    # divide the dataset intp training set, validation set and test set

    SPILT = 1 - 2 * test_ratio
    trainData = tf.data.Dataset.from_tensor_slices(
        (dataPath[:int(SPILT * len(dataLabel))], dataLabel[:int(SPILT * len(dataLabel))]))
    vaildData = tf.data.Dataset.from_tensor_slices((dataPath[int(SPILT * len(dataLabel)):(
                int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3))], dataLabel[int(SPILT * len(dataLabel)):(
                int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3))]))
    testData = tf.data.Dataset.from_tensor_slices((dataPath[
                                                   (int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3)):],
                                                   dataLabel[
                                                   (int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3)):]))

    def prePicPNG(path, label):
        temp = tf.io.read_file(path)
        temp = tf.cond(tf.image.is_jpeg(temp),
                       lambda: tf.image.decode_jpeg(temp, channels=3),
                       lambda: tf.image.decode_png(temp, channels=3))
        temp = tf.cast(temp, tf.float32)
        temp /= 255.0
        temp = tf.image.resize(temp, [dim, dim])
        return temp, label

    trainData = trainData.map(prePicPNG).shuffle(SHUFFLE).batch(BATCH)
    vaildData = vaildData.map(prePicPNG).shuffle(SHUFFLE).batch(1)
    testData = testData.map(prePicPNG).batch(1)
    return trainData, vaildData, testData

### Now load the example data to report classification result
metadir1 = './data/TB_Chest_Radiography_Database/Normal.metadata.CSV'
metadir2 = './data/TB_Chest_Radiography_Database/Tuberculosis.metadata.CSV'
img_dir1 = './data/TB_Chest_Radiography_Database/Normal/'
img_dir2 = './data/TB_Chest_Radiography_Database/Tuberculosis/'
seed = 1  # random.randint(1,100)
SHUFFLE = 1000
BATCH = 5
test_ratio = 0.2
dim = 192
trainData, validData, testData = trainData_load(metadir1, metadir2, img_dir1, img_dir2, seed,  # random.randint(1,100),
                                                SHUFFLE, BATCH, test_ratio, dim)
model = tf.keras.models.load_model('./submit_version_classification/default_model/classification_model.h5')
## predict test data
preds = model.predict(testData)
pred_result = []
for i in range(len(preds)):
    if preds[i][0] > preds[i][1]:
        pred_result.append(0)
    else:
        pred_result.append(1)
pred_result = np.array(pred_result)
## get the ground truth
testData_lab = np.concatenate([y for x, y in testData], axis=0)
# visualize the precision, recall, accuracy and f1 score
clf_report = classification_report(testData_lab, pred_result, target_names=list(["Normal", "TB"]), output_dict=True)
fig = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True).get_figure()
fig.savefig('classification_report.png')
