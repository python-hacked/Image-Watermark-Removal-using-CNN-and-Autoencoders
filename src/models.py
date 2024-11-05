import tensorflow as tf
from keras import layers, models




def create_autoencoder(inpute_shape):
#encoder 
    inpute_image  = tf.keras.Input(shape = inpute_shape)
    x = layers.Conv2D(64, (3,3),activation="relu", padding="same")(inpute_image)
    x = layers.MaxPooling2D((2,2),padding="same")(x)
    x = layers.Conv2D(32,(3,3),activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2,2),padding="same",)(x)
#dcoder
    x = layers.Conv2D(32,(3,3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(64,(3,3), activation="relu", padding="same")(x)
    x =  layers.UpSampling2D((2,2))(x)
    decode = layers.Conv2D(3,(3,3), activation="sigmoid",padding="same")(x)

    autoencoder = models.Model(inpute_image,decode)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder


