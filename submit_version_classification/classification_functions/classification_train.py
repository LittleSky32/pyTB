import matplotlib.pyplot as plt
import tensorflow as tf

def classification_train(model, patience, epoch, trainData, vaildData, testData, model_path, training_process_img_dir):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,restore_best_weights=True, min_delta=1e-4,mode='min')
    history = model.fit(trainData,shuffle=True,epochs=epoch,validation_data=vaildData, callbacks = [callback])
    # plot the loss and accuracy change
    plt.plot(history.history['accuracy'],label = 'accuracy')
    plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
    plt.plot(history.history['loss'],label = 'loss')
    plt.plot(history.history['val_loss'],label = 'val_loss')
    plt.legend(loc='best')
    plt.savefig(training_process_img_dir +'training_process.png')
    plt.show()


    # test the training
    loss, accuracy = model.evaluate(testData)
    print('--------------------training over--------------------\n\n')
    print('***********************model evaluation************************')
    print('The loss of the best trained model is '+ str(loss)+'; the accuracy of the best trained model is '+str(accuracy))
    # save the model
    model_dir = model_path+'classification.h5'
    model.save(model_dir)
    print('The trained model has been saved in '+model_path)
    return model, history