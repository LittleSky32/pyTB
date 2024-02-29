### this file is used to create UI train for our software using click
# import module
import click
import sys
import os
sys.path.append('./classification_functions/')
import dataLoad
import classification_model
import classification_train
if not os.path.exists('./models_saved/'):
    os.makedirs('./models_saved/')
if not os.path.exists('./train_output/'):
    os.makedirs('./train_output/')

# set the parameters
@click.command()
@click.option('--metadir1', required = True, help='The path where the metadata file of the type1 saved. (NEED .CSV FILES!)')
@click.option('--metadir2', required = True, help='The path where the metadata file of the type2 saved. (NEED .CSV FILES!)')

@click.option('--img_dir1', required = True, help='The path where type1 figures saved.')
@click.option('--img_dir2', required = True, help='The path where type2 figures saved.')

@click.option('--seed', default = 1, help='The random seed.')
@click.option('--shuffle', default=1000, help='The shuffle number of the dataset input.')
@click.option('--batch', default=16, help='The batch size of the training.')
@click.option('--test_ratio', default=0.2, help='The percentage of the test dataset in the whole dataset. The validation dataset is the same as test dataset.')
@click.option('--dim', default=192, help='The reshaping size of the figures for training input.')
@click.option('--patience', default=10, help='The length of epoch waiting for when the validation loss stop decreasing.')
@click.option('--epoch', default=50, help='The epoch of training.')

@click.option('--model_path', default='./models_saved/', help='The path for saving the trained model.')
@click.option('--training_process_img_dir', default='./train_output/', help='The path for saving training output.')



def train(metadir1,metadir2,img_dir1,img_dir2,seed, shuffle,batch,test_ratio,dim, patience, epoch, model_path, training_process_img_dir):
    click.echo('========================================Dataload start==================================================')
    click.echo('========================================================================================================')

    trainData, vaildData, testData = dataLoad.trainData_load(metadir1, metadir2, img_dir1, img_dir2, seed, shuffle, batch, test_ratio, dim)

    click.echo('=======================================Dataload finished================================================')
    click.echo('========================================================================================================')
    click.echo('==========================================Train start===================================================')
    click.echo('========================================================================================================')
    model = classification_model.incepLeNet((dim,dim,3))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    classification_train.classification_train(model, patience, epoch, trainData, vaildData, testData, model_path, training_process_img_dir)

    click.echo('===========================================Train start==================================================')
    click.echo('========================================================================================================')
if __name__ == '__main__':
    train()
