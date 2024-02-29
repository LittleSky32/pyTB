### this file is used to create UI training for our software using click
# import the module
import click
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'# shield the warnings about GPU compability of the tensorflow
import sys
sys.path.append('./segmentation_functions/')# add the module path
import dataLoad
if not os.path.exists('./models_saved/'):
    os.makedirs('./models_saved/')
if not os.path.exists('./train_output/'):
    os.makedirs('./train_output/')
import segmentation_model
import segmentation_train

# set the parameters of the terminal software
@click.command()
@click.option('--img_dir', required=True, help='The path of the figures which is needed for training segmention.')
@click.option('--mask_dir', required=True, help='The path of the masks corresponded to the figures.')
@click.option('--seed', default=1, help='The random seed for reproducing the result.')
@click.option('--dim', default=256, help='The size to reshape the input figures.')
@click.option('--val_size', default=0.1, help='The percent of the validation set in the whole data set.')
@click.option('--model_path', default='./models_saved/', help='The path for saving the trained model.')
@click.option('--patience', default=10, help='The length of epoch waiting for when the validation loss stop decreasing.')
@click.option('--learnrate', default=0.0001, help='The learning rate of the training.')
@click.option('--batch_size',default=16, help='The size of batch.')
@click.option('--epoch', default=50, help='The epoch of training.')
@click.option('--train_output_path', default='./train_output/', help='The path for saving training output.')


# run the training function
def train(img_dir, mask_dir, seed, dim, val_size, model_path,patience,learnrate,batch_size,epoch,train_output_path):
    click.echo('========================================Dataload start==================================================')
    click.echo('========================================================================================================')

    # load the data and divided into two datasets for training and validation.
    train_vol, train_seg, test_vol, test_seg = dataLoad.trainData_load(img_dir, mask_dir, seed, dim, val_size)
    click.echo('=======================================Dataload finished================================================')
    click.echo('========================================================================================================')
    click.echo('==========================================Train start===================================================')
    click.echo('========================================================================================================')
    
    # Train the dataset
    model = segmentation_model.segmentation_model(dim)
    segmentation_train.segmentation_train(model_path, patience,learnrate,model,train_vol,train_seg,
    batch_size,epoch,test_vol,test_seg,train_output_path)
    click.echo('=========================================Train finished=================================================')
    click.echo('========================================================================================================')





if __name__ == '__main__':
    train()
