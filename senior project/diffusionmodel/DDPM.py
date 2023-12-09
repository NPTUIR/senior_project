import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
from utils.metrics import KID
from utils import CustomLayer
from utils.setpath import Datapath
from utils.sketchdata import DataLoader
from tensorflow.keras.utils import plot_model

class DiffusionModel(keras.Model):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.path = Datapath()
        config.save_to_json(path=f'{self.path.LOG_CONFIGPATH}/config.json')
        self.normalizer = layers.Normalization()
        self.network = CustomLayer.get_network(config.image_size)
        self.ema_network = keras.models.clone_model(self.network)
        plot_model(model=self.network, to_file=f'{self.path.LOG_CONFIGPATH}/model.png',show_shapes=True)

        self.compile(optimizer=AdamW(learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay), loss=keras.losses.mean_absolute_error)
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid",image_size=self.config.image_size, kid_image_size=self.config.kid_image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        images = self.normalizer.mean[:,:,:,:3] + images * self.normalizer.variance[:,:,:,:3]**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.acos(self.config.max_signal_rate)
        end_angle = tf.acos(self.config.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, sketch_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps, sketch_images):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, sketch_images, noise_rates, signal_rates, training=False)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

        return pred_images

    def generate(self, num_images, diffusion_steps, sketch_images):
        initial_noise = tf.random.normal(shape=(num_images, self.config.image_size, self.config.image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps, sketch_images)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        sketch_images = images[:, :, :, :3]
        images = images[:, :, :, 3:]
        noises = tf.random.normal(shape=(self.config.batch_size, self.config.image_size, self.config.image_size, 3))

        diffusion_times = tf.random.uniform(shape=(self.config.batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        # noisy_sketch = signal_rates * sketch_images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(noisy_images, sketch_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.config.ema * ema_weight + (1 - self.config.ema) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        images = self.normalizer(images, training=False)#(1, 1, None, 256, 256, 6)
        images = tf.squeeze(images, axis=[0, 1])#(None, 256, 256, 6)
        images = tf.reshape(images, [self.config.batch_size, self.config.image_size, self.config.image_size, 6])

        sketch_images = images[:, :, :, :3]#(None, 256, 256, 3)
        image = images[:, :, :, 3:]

        noises = tf.random.normal(shape=(self.config.batch_size, self.config.image_size, self.config.image_size, 3))
        diffusion_times = tf.random.uniform(
            shape=(self.config.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * image + noise_rates * noises

        pred_noises, pred_images = self.denoise(
            noisy_images, sketch_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(image, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        image = self.denormalize(image)

        generated_images = self.generate(
            num_images=self.config.batch_size, diffusion_steps=self.config.kid_diffusion_steps, sketch_images=sketch_images)

        self.kid.update_state(image, generated_images)
        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=1, num_cols=2, path=''):
        if path=='':
            path = f'{self.path.RESULT_IMGPATH}/{int(time.time())}.png'
        idx = np.random.randint(0, 400, size=(self.config.batch_size, ))
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=self.config.plot_diffusion_steps,
            sketch_images=self.sketchdata[idx]
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(path)
        # plt.close()

    def set_callback(self):
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.path.DIFFUSION_PATH,
            save_weights_only=True,
            monitor="val_kid",
            mode="min",
            save_best_only=True,
        )
    def train(self):
        self.sketchdata = np.load('./data/64x64/sketch64.npy')
        realdata = np.load('./data/64x64/real64.npy')

        data = np.append(self.sketchdata, realdata, axis=-1)
        data = np.tile(data, (5, 1, 1, 1))
        np.random.shuffle(data)
        # plt.imshow(data[0,:,:,3:])
        # plt.show()

        # self.normalizer.adapt(data)
        self.test_sketchdata = np.load('./data/64x64/test_sketch.npy')
        test_realdata = np.load('./data/64x64/test_real64.npy')

        test_data = np.append(self.test_sketchdata, test_realdata, axis=-1)
        self.history = self.fit(data, epochs=self.config.num_epochs, batch_size=self.config.batch_size,
            validation_data=[test_data],
            callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=self.plot_images),
                self.checkpoint_callback])
        self.network.save(f'{self.path.logpath}/CDDPM.h5')
        self.save_loss()
    def save_loss(self):
        plt.figure(figsize=(8, 6))
        plt.title('Training Loss',fontsize=20)
        plt.xlabel('Epochs',fontsize=17)
        plt.ylabel('Loss',fontsize=17)
        for key in self.history.history:
            plt.plot(self.history.history[key], label=key)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(f'{self.path.LOG_LOSSPATH}/loss.png')
        for key in self.history.history:
            np.save(arr=self.history.history[key], file=f'{self.path.LOG_LOSSPATH}/{key}.npy')
    def result(self):
        path = f'{self.path.LOG_PREDICTPATH}/predict.png'
        self.load_weights(self.path.DIFFUSION_PATH)
        self.plot_images(path=path)