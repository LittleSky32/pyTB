import tensorflow as tf
import os
import pandas as pd


# predict the classification
def classification_predict(dim: int, model_dir: str, pred_dir: str, pred_result_path: str):
    """

    Args:
        dim: weight and height of the image (weight = height = dim)
        model_dir: the directory of model
        pred_dir: the directory for images to be predicted
        pred_result_path: the directory to save prediction results

    Returns:
        None
    """
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

    # read in the images and their labels for prediction
    predPath = os.listdir(os.path.join(pred_dir))
    pred_dataPath = [pred_dir + pred_path for pred_path in predPath]
    pred_dataLabel = [0] * len(predPath)
    predData = tf.data.Dataset.from_tensor_slices((pred_dataPath, pred_dataLabel)).map(prePicPNG).batch(1)
    # load model
    model = tf.keras.models.load_model(model_dir)
    pred_pro = model.predict(predData)
    pred_result = []
    for i in range(len(pred_pro)):
        # decide the type of image based on the 2 output from our network
        if pred_pro[i][0] > pred_pro[i][1]:
            pred_result.append(['Normal', predPath[i]])
        else:
            pred_result.append(['Diseased', predPath[i]])
    # save the result of prediction
    pred_result_df = pd.DataFrame(data=pred_result, columns=['predict type', 'file name'])
    pred_result_df.to_csv(pred_result_path + 'predict_result.csv', index=0)
    print('Predict result is saved in ' + pred_result_path)
    return
