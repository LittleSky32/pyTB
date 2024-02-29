### this file is used to create UI prediction for our software using click
# import module
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import click
import sys
sys.path.append('./classification_functions/')
import classification_predict
if not os.path.exists('./pred_output/'):
    os.makedirs('./pred_output/')


# set parameters
@click.command()
@click.option('--dim', default=192, help='The reshaping size of the figures for predicting.')
@click.option('--model_path', default='./default_model/classification_model.h5', help='The path of the model for predicting.')
@click.option('--pred_dir', required=True, help='The path of the figures for predicting.')
@click.option('--pred_result_path', default='./pred_output/', help='The path of prediction result saved.')

# link to click
def predict(dim, model_path, pred_dir, pred_result_path):
    classification_predict.classification_predict(dim, model_path, pred_dir, pred_result_path)

if __name__ == '__main__':
    predict()