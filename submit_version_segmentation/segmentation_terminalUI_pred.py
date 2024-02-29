### this file is used to create UI prediction for our software using click
# import module
import click
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'# shield the warnings about GPU compability of the tensorflow
import sys
sys.path.append('./segmentation_functions/')# add the module path
import segmentation_predict
if not os.path.exists('./pred_output/'):# create the default path
    os.makedirs('./pred_output/')
# set the parameters of the terminal software
@click.command()
@click.option('--pred_dir', required=True, help='The path of the figures need to be predicted for masking.')
@click.option('--model_path', default='./default_model/segmentation_model.h5', help='The path of the model used for prediction.')
@click.option('--pred_output_path', default='./pred_output/', help='The path of the predicted output.')
@click.option('--dim', default=256, help='The reshape size of the input figrues for predicting.')

# run the prediction model
def predict(pred_dir, model_path, pred_output_path, dim):
    segmentation_predict.segmentation_pred(pred_dir, model_path, pred_output_path, dim)


if __name__ == '__main__':
    predict()
