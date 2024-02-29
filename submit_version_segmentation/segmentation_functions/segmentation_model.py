import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred):
    """
    define the dice coefficient function
    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by the model

    Returns:
    dice loss
    """
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + 1) / (K.sum(y_truef) + K.sum(y_predf) + 1)
# dice loss is just 1-dice_coef
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def segmentation_model(dim=256):
    """
    define U-Net model
    Args:
        dim: weight = height = dim

    Returns:
    unet model
    """
    def uconv2d(conv_size: int, input):
        convn = tf.keras.layers.Conv2D(conv_size, (3, 3), activation='relu', padding='same')(
            tf.keras.layers.Conv2D(conv_size, (3, 3), activation='relu', padding='same')(input))
        return convn

    def unet(input_size):
        inputs = tf.keras.Input(input_size)

        conv1 = uconv2d(16, inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = uconv2d(32, pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = uconv2d(64, pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = uconv2d(128, pool3)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = uconv2d(256, pool4)

        up6 = tf.keras.layers.concatenate(
            [tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = uconv2d(128, up6)

        up7 = tf.keras.layers.concatenate(
            [tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = uconv2d(64, up7)

        up8 = tf.keras.layers.concatenate(
            [tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = uconv2d(32, up8)

        up9 = tf.keras.layers.concatenate(
            [tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = uconv2d(16, up9)

        conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        return tf.keras.Model(inputs=[inputs], outputs=[conv10])

    # compile the model, using dice loss defined above
    model = unet(input_size=(dim, dim, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss=dice_coef_loss,
                  metrics=['binary_accuracy', 'accuracy'])

    return model
