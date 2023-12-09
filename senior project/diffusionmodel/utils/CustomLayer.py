import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def sinusoidal_embedding(x,embedding_dims=64,embedding_max_frequency=1000.0):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2))
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    # sketch_input = keras.Input(shape=(image_size, image_size, 3))
    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)
    x = layers.Conv2D(32, kernel_size=1)(noisy_images) #64
    # s = layers.Conv2D(32, kernel_size=1)(sketch_input)
    # x = layers.Multiply()([n, s])
    x = layers.Concatenate()([x, e])
    skips = []
    x = DownBlock(64, 2)([x, skips]) #32
    x = DownBlock(128, 2)([x, skips]) #16
    x = DownBlock(256, 2)([x, skips]) #8
    x = DownBlock(256, 2)([x, skips]) #4
    x = ResidualBlock(512)(x) #2
    x = ResidualBlock(512)(x) #1
    x = UpBlock(256, 2)([x, skips]) #2
    x = UpBlock(256, 2)([x, skips])
    x = UpBlock(128, 2)([x, skips])
    x = UpBlock(64, 2)([x, skips])
    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)
    model = keras.Model([noisy_images, noise_variances], x, name="residual_unet")
    # from tensorflow.keras.utils import plot_model
    # plot_model(model,'./model.png',show_shapes=True)
    return model

# get_network(64)