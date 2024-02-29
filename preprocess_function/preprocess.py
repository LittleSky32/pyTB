# import the module
import click
import cv2
import os
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm


# set the parameters
@click.command()
@click.option('--original_path', required=True, help='The path of figures which need to be preprocessed')
@click.option('--preprocessed_path', required=True, help='The path where the processed data to save.')
def preprocess(original_path: str, preprocessed_path: str):
    """
    preprocess images using CLAHE, turn all of them into
    Args:
        original_path: path of original images to read in
        preprocessed_path: path of preprocessed images to be saved

    Returns:
        None
    """
    # load the original figures
    print('Loading figures......')
    fig_name = os.listdir(original_path)
    # transform figures into gray channel
    for i in tqdm(fig_name):
        img = cv2.imread(original_path + str(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        cv2.imwrite(preprocessed_path + str(i), img2)

    def clahe(img):
        """
        Use CLAHE to augment the image
        Args:
            img: image to imput

        Returns:
            cle: image after CLAHE
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cle = clahe.apply(img)
        return cle

    img_names = os.listdir(preprocessed_path)
    print('Start processing!')
    for img_name in tqdm(img_names):
        img_path = os.path.join(preprocessed_path, img_name)
        img = imageio.imread(img_path)
        result = clahe(img)  # preprocess the figures by Contrast Limited Adaptive HistogramEqualization (CLAHE) method
        save_path = os.path.join(preprocessed_path, img_name)
        cv2.imwrite(save_path, result)
    print('Preprocessing finished!')
    return


if __name__ == '__main__':
    preprocess()
