import tensorflow as tf
from tensorflow.keras import layers


# induce inception_V3
def Inception(x, n=1):
    """Inception model

    Args:
        x: input preprocessed images
        n: number to control filters in the model, default is 1
    """
    Inception1 = layers.Conv2D(n * 2, (1, 1), padding='SAME', activation='relu')(x)
    Inception1 = layers.Conv2D(n * 8, (3, 3), padding='SAME', activation='relu')(Inception1)
    Inception1_1 = layers.Conv2D(n * 8, (1, 3), padding='SAME', activation='relu')(Inception1)
    Inception1_2 = layers.Conv2D(n * 8, (3, 1), padding='SAME', activation='relu')(Inception1)

    Inception2 = layers.Conv2D(n * 4, (1, 1), padding='SAME', activation='relu')(x)
    Inception2_1 = layers.Conv2D(n * 32, (1, 3), padding='SAME', activation='relu')(Inception2)
    Inception2_2 = layers.Conv2D(n * 32, (3, 1), padding='SAME', activation='relu')(Inception2)

    Inception3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='SAME')(x)
    Inception3 = layers.Conv2D(n * 4, (1, 1), padding='SAME', activation='relu')(Inception3)

    Inception4 = layers.Conv2D(n * 16, (1, 1), padding='SAME', activation='relu')(x)

    return layers.Concatenate()([Inception1_1, Inception1_2, Inception2_1, Inception2_2, Inception3, Inception4])


# mix the inception_V3 with original LeNet-5
def incepLeNet(input_size):
    inputs = tf.keras.Input(input_size)
    incpt1 = Inception(inputs)
    pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='SAME')(incpt1)

    incpt2 = Inception(pool1, 2)
    pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='SAME')(incpt2)

    incpt3 = Inception(pool2, 2)
    pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='SAME')(incpt3)

    incpt4 = Inception(pool3, 3)
    pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='SAME')(incpt4)

    incpt5 = Inception(pool4, 3)
    pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='SAME')(incpt5)

    incpt6 = Inception(pool5, 4)
    pool6 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='SAME')(incpt6)
    flatten1 = layers.Flatten()(pool6)
    dense1 = layers.Dense(1024, activation='relu')(flatten1)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    dropout1 = tf.keras.layers.Dropout(0.5)(dense2)
    dense3 = layers.Dense(2, activation='softmax')(dropout1)

    return tf.keras.Model(inputs=[inputs], outputs=dense3)


# original version of the model referring to LeNet-5
'''def classification_model(input_size:int):
    model = Sequential()
    model.add(layers.Conv2D(32,(5,5),padding = 'SAME',activation='relu',input_shape=[input_size,input_size,3]))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,32),padding = 'SAME',activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2,activation='softmax'))



    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    return model'''
