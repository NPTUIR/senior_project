import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LeakyReLU, Activation, Conv2D, Conv2DTranspose, Dropout, Concatenate, ZeroPadding2D, Add
from tensorflow_addons.layers import InstanceNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from utils.data import DataLoader
from utils.setpath import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy, MAE
from tensorflow.keras.initializers import RandomNormal, Zeros
import utils.img2video as video
import  tensorflow as tf
import time

class CustomSchedule(LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        super(CustomSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        drop = self.initial_learning_rate / 100
        epochs_drop = 400
        return tf.cond(
            tf.math.greater(step, 40000),
            lambda: self.initial_learning_rate,
            lambda: self.initial_learning_rate - (step // epochs_drop - 100) * drop)


class Cycle_GAN:
    def __init__(self, model_name=None, config=None):
        if model_name == None:
            self.modelname = self.__class__.__name__
        else:
            self.modelname = model_name
        self.path = Path(model_name=model_name)
        self.config = config
        self.lambda_cycle = 10
        self.lambda_id = 0.5*self.lambda_cycle
        self.lr = 0.0002
        # self.binary_crossentropy_loss = BinaryCrossentropy(from_logits=True)

        self.genModel_sketch2real = self.build_generator()
        self.genModel_real2sketch = self.build_generator()
        self.disModel_sketch = self.build_discriminator()
        self.disModel_real = self.build_discriminator()

        # lr_scheduler = LearningRateScheduler(self.step_decay)
        self.advModel = self.build_adversialmodel()
        self.disc_patch = (16, 16, 1)
        self.sketchdata, self.realdata = DataLoader().load_data(normalization='[-1,1]')
        test_sketchdata, test_realdata = DataLoader().load_testing_data()

        np.save(file='test_sketch.npy', arr=test_sketchdata)
        np.save(file='test_real.npy', arr=test_realdata)

    def ResidualBlock(self, units, inputs):
        r1 = Conv2D(units, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(inputs)
        r2 = Conv2D(units, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(r1)
        out = Add()([inputs, r2])
        return out
    def build_generator(self):
        input_ = Input(shape=(256, 256, 3))
        # c
        c1 = Conv2D(64, kernel_size=(7, 7), strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(input_)
        c1 = InstanceNormalization()(c1)
        c1 = Activation('relu')(c1)
        # dk
        d2 = Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(c1)
        d2 = InstanceNormalization()(d2)
        d2 = Activation('relu')(d2)
        d3 = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(d2)
        d3 = InstanceNormalization()(d3)
        d3 = Activation('relu')(d3)
        # Rk
        r1 = self.ResidualBlock(256, d3)
        r2 = self.ResidualBlock(256, r1)
        r3 = self.ResidualBlock(256, r2)
        r4 = self.ResidualBlock(256, r3)
        r5 = self.ResidualBlock(256, r4)
        r6 = self.ResidualBlock(256, r5)
        r7 = self.ResidualBlock(256, r6)
        r8 = self.ResidualBlock(256, r7)
        r9 = self.ResidualBlock(256, r8)
        # uk
        u1 = Conv2DTranspose(128, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(r9)
        u1 = InstanceNormalization()(u1)
        u1 = Activation('relu')(u1)
        u2 = Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(u1)
        u2 = InstanceNormalization()(u2)
        u2 = Activation('relu')(u2)
        out = Conv2DTranspose(3, kernel_size=(7, 7), strides=1, padding='same', activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(u2)

        model = Model(input_, out)
        plot_model(model, 'G model.png', show_shapes=True)
        return model

    def build_discriminator(self):
        input_ = Input(shape=(256, 256, 3))
        c1 = Conv2D(64, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(input_)
        c1 = LeakyReLU(0.2)(c1)
        c2 = Conv2D(128, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(c1)
        c2 = InstanceNormalization()(c2)
        c2 = LeakyReLU(0.2)(c2)
        c3 = Conv2D(256, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(c2)
        c3 = InstanceNormalization()(c3)
        c3 = LeakyReLU(0.2)(c3)
        c4 = Conv2D(512, kernel_size=(4, 4), strides=2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(c3)
        c4 = InstanceNormalization()(c4)
        c4 = LeakyReLU(0.2)(c4)
        out = Conv2D(1, kernel_size=(4, 4), strides=1, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=None))(c4)

        model = Model(input_, out)
        optimizer = Adam(learning_rate=CustomSchedule(initial_learning_rate=self.lr), beta_1=0.5)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        plot_model(model, 'D model.png', show_shapes=True)
        return model
    def build_adversialmodel(self):
        img_sketch = Input(shape=(256, 256, 3))
        img_real = Input(shape=(256, 256, 3))

        fake_real = self.genModel_sketch2real(img_sketch)
        fake_sketch = self.genModel_real2sketch(img_real)

        reconstruction_sketch = self.genModel_real2sketch(fake_real)
        reconstruction_real = self.genModel_sketch2real(fake_sketch)

        img_sketch_identity = self.genModel_real2sketch(img_sketch)
        img_real_identity = self.genModel_sketch2real(img_real)

        self.disModel_sketch.trainable = False
        self.disModel_real.trainable = False
        valid_sketch = self.disModel_sketch(fake_sketch)
        valid_real = self.disModel_real(fake_real)

        model = Model(inputs=[img_sketch, img_real],
                      outputs=[valid_sketch, valid_real,
                               reconstruction_sketch, reconstruction_real,
                               img_sketch_identity, img_real_identity])

        optimizer = Adam(learning_rate=CustomSchedule(initial_learning_rate=self.lr), beta_1=0.5)
        model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                      loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                      optimizer=optimizer)
        return model

    def train(self, epochs, batch_size=1, sample_interval=50):
        valid = np.ones((batch_size, ) + self.disc_patch)
        fake = np.zeros((batch_size, ) + self.disc_patch)
        for epoch in range(epochs):
            idx = np.random.randint(0, self.sketchdata.shape[0], batch_size)
            real_images = self.realdata[idx]
            real_sketch = self.sketchdata[idx]
            fake_real = self.genModel_sketch2real.predict(real_sketch)
            fake_sketch = self.genModel_real2sketch.predict(real_images)

            d_sketch_loss_real = self.disModel_sketch.train_on_batch(real_sketch, valid)
            d_sketch_loss_fake = self.disModel_sketch.train_on_batch(fake_sketch, fake)
            d_sketch_loss = 0.5 * np.add(d_sketch_loss_real, d_sketch_loss_fake)

            d_realimg_loss_real = self.disModel_real.train_on_batch(real_images, valid)
            d_realimg_loss_fake = self.disModel_real.train_on_batch(fake_real, fake)
            d_realimg_loss = 0.5 * np.add(d_realimg_loss_real, d_realimg_loss_fake)
            d_loss = 0.5 * np.add(d_sketch_loss, d_realimg_loss)

            g_loss = self.advModel.train_on_batch([real_sketch, real_images],
                                                  [valid, valid, real_sketch, real_images, real_sketch, real_images])
            if epoch==0:
                dloss = d_loss
                gloss = g_loss
            else:
                dloss = np.append(dloss, d_loss, axis=0)
                gloss = np.append(gloss, g_loss, axis=0)
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss[0]}, acc: {100 * d_loss[1]:.3f}] [G loss: {g_loss[0]:.5f}, adv: {np.mean(g_loss[1:3]):.5f}, recon: {np.mean(g_loss[3:5]):.5f}, id: {np.mean(g_loss[5:6]):.5f}]")

            if (epoch + 1) % 500 == 0 or epoch == 0 or (epoch + 1 <= 3000 and (epoch + 1) % 40 == 0):
                self.predict(epoch=epoch)

        np.save(file='dloss.npy', arr=dloss)
        np.save(file='gloss.npy', arr=gloss)
        self.genModel_sketch2real.save(f'cyclegan/generator.h5')

    def predict(self, num_images=3, training=True, **kwargs):
        if training:
            idx = np.array([10, 20, 30])  # 選擇第11、21、31張圖片，雖然沒有意義但統一比較好~
            fake_images = self.genModel_sketch2real.predict(self.sketchdata[idx])  # 生成器生成圖片
            fake_images = ((fake_images + 1) / 2)  # 反正規化。我用tanh所以要這樣做，用sigmoid輸出則不用反正規化
            plt.figure(figsize=(8, 8))
            plt.title(f'Training Process of {kwargs["epoch"] + 1} Epochs')
            for j in range(num_images):
                plt.subplot(num_images, num_images, num_images * j + 1)
                plt.title('Input Sketch Image', fontsize=12)
                plt.imshow(self.sketchdata[idx][j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j + 2)
                plt.title('Generated Image', fontsize=12)
                plt.imshow(fake_images[j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j + 3)
                plt.title('Ground Truth', fontsize=12)
                plt.imshow((self.realdata[idx][j] + 1) / 2)
                plt.axis("off")
            plt.savefig(f'cyclegan/img/epoch{kwargs["epoch"] + 1:0>5}.png')  # 儲存圖片 kwargs["epoch"] + 1:0>5這邊不要改，否則保證之後影片生成時你會哭
        else:
            test_sketchdata, test_realdata = DataLoader().load_testing_data()
            idx = np.array([10, 20, 30])
            # model = load_model(f'{self.path.MODELPATH}/{self.modelname}_generator.h5')  # 下載生成模型，或者直接使用都可以
            generated_images = self.genModel_sketch2real.predict(test_sketchdata[idx])
            generated_images = ((generated_images + 1) / 2)  # 反正規化。我用tanh所以要這樣做，用sigmoid輸出則不用反正規化
            plt.figure(figsize=(8, 8))
            plt.title(f'predict')
            plt.suptitle(f'The Images of {self.modelname} Generate',
                         fontsize=17)  # {self.modelname}可以改成你的模型名稱例如CVAE、CGAN、Cycle-GAN
            for j in range(num_images):  # 這邊都不要動
                plt.subplot(num_images, num_images, num_images * j + 1)
                plt.title('Input Sketch Image', fontsize=12)
                plt.imshow(test_sketchdata[j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j + 2)
                plt.title('Generated Image', fontsize=12)
                plt.imshow(generated_images[j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j + 3)
                plt.title('Ground Truth', fontsize=12)
                plt.imshow(test_realdata[j])
                plt.axis("off")
            plt.savefig(f'cyclegan/img/predict.png')  # 儲存圖片



if __name__=='__main__':
    cyclegan = Cycle_GAN()
