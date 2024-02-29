import pandas as pd
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf


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

    nor_data = pd.read_csv(metadir1)
    tb_data = pd.read_csv(metadir2)

    nor_dataPath = [img_dir1 + nor_path + '.png' for nor_path in nor_data.iloc[:, 0]]
    nor_dataLabel = [0] * len(nor_dataPath)

    tb_dataPath = [img_dir2 + unor_path + '.png' for unor_path in tb_data.iloc[:, 0]]
    tb_dataLabel = [1] * len(tb_dataPath)

    # return the size of two size
    print('The size of type 1 dataset is ' + str(len(nor_dataPath)))
    print('The size of type 2 dataset is ' + str(len(tb_dataPath)))

    # equalize the size of two datasets
    maxlen = min(len(nor_dataPath), len(tb_dataPath))
    # combine the paths
    dataPath = nor_dataPath[0:maxlen] + tb_dataPath[0:maxlen]
    dataLabel = nor_dataLabel[0:maxlen] + tb_dataLabel[0:maxlen]

    # shuffle the datasets
    random.seed(seed)
    random.shuffle(dataPath)
    random.shuffle(dataLabel)
    dataPath = dataPath
    dataLabel = dataLabel

    # split the datasets into training set, validation set and testing set.
    SPILT = 1 - 2 * test_ratio
    trainData = tf.data.Dataset.from_tensor_slices(
        (dataPath[:int(SPILT * len(dataLabel))], dataLabel[:int(SPILT * len(dataLabel))]))
    testData = tf.data.Dataset.from_tensor_slices((dataPath[int(SPILT * len(dataLabel)):(
                int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3))], dataLabel[int(SPILT * len(dataLabel)):(
                int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3))]))
    vaildData = tf.data.Dataset.from_tensor_slices((dataPath[
                                                    (int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3)):],
                                                    dataLabel[
                                                    (int(SPILT * len(dataLabel)) + int(SPILT * len(dataLabel) / 3)):]))

    print('The size of train dataset dataset is ' + str(len(trainData)))
    print('The size of test dataset dataset is ' + str(len(testData)))
    print('The size of validation dataset dataset is ' + str(len(vaildData)))

    def prePicPNG(path, label):
        """
        load image, rescale and resize it
        Args:
            path: the path of images
            label: the label of images

        Returns:
        preprocessed image and corresponding labels
        """
        temp = tf.io.read_file(path)
        temp = tf.cond(tf.image.is_jpeg(temp),
                       lambda: tf.image.decode_jpeg(temp, channels=3),
                       lambda: tf.image.decode_png(temp, channels=3))
        temp = tf.cast(temp, tf.float32)
        temp /= 255.0
        temp = tf.image.resize(temp, [dim, dim])
        return temp, label

    # use the functions above to split original data into training,
    # validation and test with shuffle and batch defined above
    trainData = trainData.map(prePicPNG).shuffle(SHUFFLE).batch(BATCH)
    testData = testData.map(prePicPNG).shuffle(SHUFFLE).batch(1)
    vaildData = vaildData.map(prePicPNG).shuffle(SHUFFLE).batch(1)
    return trainData, testData, vaildData
