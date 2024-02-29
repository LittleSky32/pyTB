from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('.')
import segmentation_model


def segmentation_train(model_path, patience, learnRate, model, train_vol, train_seg,
                       batchSize, epoch, validation_vol, validation_seg, train_output_path):
    """
    Train the segmentation network
    Args:
        model_path: path where the model is saved
        patience: Number of epochs with no improvement after which training will be stopped.
        learnRate: learning rate
        model: keras model used for training
        train_vol: training features
        train_seg: training truth
        batchSize: batch size
        epoch: number of epochs
        validation_vol: validation features
        validation_seg: validation truth
        train_output_path: path where training output is saved

    Returns:

    """
    # set loss functions and checkpoint for the model

    callbacks_list = [ModelCheckpoint(filepath=(model_path + 'model_weights.ckpt'), monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='auto', save_weights_only=True),
                      EarlyStopping(monitor="val_loss", mode="auto", patience=patience),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto',
                                        min_delta=learnRate, cooldown=2, min_lr=1e-6)]

    # Train the model
    history = model.fit(x=train_vol,
                        y=train_seg,
                        batch_size=batchSize,
                        epochs=epoch,
                        validation_data=(validation_vol, validation_seg),
                        callbacks=callbacks_list)
    # Save the model and its weights
    model.save_weights(model_path + 'model_weights.ckpt', save_format='tf')
    model.save(model_path + 'segmentation_model.h5')
    print('The weights are saved at ' + model_path)
    print('The model is saved at ' + model_path)
    # plot the loss plot and accuracy plot
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["binary_accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_binary_accuracy"]
    # loss plot over epoch
    plt.figure()
    plt.plot(range(epoch), train_loss, label='train_loss')
    plt.plot(range(epoch), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(train_output_path + 'loss_figures.png')

    # accuracy plot over epoch
    plt.figure()
    plt.plot(range(epoch), train_accuracy, label='train_accuracy')
    plt.plot(range(epoch), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(train_output_path + 'accuracy_figures.png')
    plt.show()

    ## example output
    pred_candidates = np.random.randint(1, validation_vol.shape[0], 10)
    preds = model.predict(validation_vol)

    plt.figure(figsize=(10, 10))

    for i in range(0, 9, 3):
        plt.subplot(3, 3, i + 1)

        plt.imshow(np.squeeze(validation_vol[pred_candidates[i]]))
        plt.title("Base Image")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 3, i + 2)
        plt.imshow(np.squeeze(validation_seg[pred_candidates[i]]))
        plt.title("Mask")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 3, i + 3)
        plt.imshow(np.squeeze(preds[pred_candidates[i]]))
        plt.title("Pridiction")
        plt.xticks([])
        plt.yticks([])
    plt.savefig(train_output_path + 'predict.png')
    return
