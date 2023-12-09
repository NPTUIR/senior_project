import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Activation, Conv2D, Conv2DTranspose, Flatten, Reshape, \
    BatchNormalization, Embedding, multiply, Dropout, Concatenate, ZeroPadding2D, Lambda
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from utils.data import DataLoader
from utils.setpath import Path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy, MAE
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.losses import mean_squared_error
import utils.img2video as video
import time
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

class SaveImageCallback(Callback):
    def __init__(self, save_path, decoder):
        # 呼叫父類別的初始化函數
        super().__init__()
        # 將period儲存為類別的屬性
        self.save_path = save_path
        self.decoder = decoder
        self.test_sketchdata, self.test_realdata = DataLoader().load_testing_data()
        self.idx = np.array([10, 20, 30])
        self.test_sketchdata = self.test_sketchdata[self.idx]
        self.test_realdata = self.test_realdata[self.idx]

    def decoder_generated(self):
        noise = np.random.random(size=(3, 1, 1, 1024))
        d_fake_images = self.decoder.predict([self.test_sketchdata, noise])
        d_fake_images = ((d_fake_images + 1) / 2)
        return d_fake_images
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            fake_images = self.model.predict([self.test_sketchdata, self.test_realdata])
            fake_images = ((fake_images + 1) / 2)
            d_fake_images = self.decoder_generated()
            plt.figure(figsize=(8, 8))
            plt.title(f'Training Process of {epoch} Epochs')
            for j in range(3):
                plt.subplot(3, 4, 4 * j + 1)
                plt.title('sketch', fontsize=12)
                plt.imshow(self.test_sketchdata[j])
                plt.axis("off")

                plt.subplot(3, 4, 4 * j + 2)
                plt.title('CVAE generated', fontsize=12)
                plt.imshow(fake_images[j])
                plt.axis("off")

                plt.subplot(3, 4, 4 * j + 3)
                plt.title('Decoder generated', fontsize=12)
                plt.imshow(d_fake_images[j])
                plt.axis("off")

                plt.subplot(3, 4, 4 * j + 4)
                plt.title('true', fontsize=12)
                plt.imshow(self.test_realdata[j])
                plt.axis("off")
            plt.savefig(f'{self.save_path}/epoch{epoch:0>5}.png')

class CVAE:
    def __init__(self,model_name=None, config=None):
        if model_name == None:
            self.modelname = self.__class__.__name__
        else:
            self.modelname = model_name

        self.path = Path(model_name=model_name)
        self.config = config
        self.Encoder = self.build_Encoder()
        # self.Encoder_sketch = self.build_Encoder()
        self.Decoder = self.build_Decoder()
        self.CVAEmodel = self.build_CVAE()
        self.sketchdata, self.realdata = DataLoader().load_data(normalization='[-1,1]')

        self.save_image_callback = SaveImageCallback(self.path.TRAININGIMGPATH, self.Decoder)
    def build_Encoder(self, f=None):
        def encoder_block(input_, unit, kernel=4, strides=2, padding='same', bn=True):
            x = Conv2D(unit, kernel_size=kernel, strides=strides, padding=padding)(input_)
            if bn:
                x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x
        if f==None:
            f = self.config.e_first_unit
        sketch_input = Input(shape=(256, 256, 3))
        real_input = Input(shape=(256, 256, 3))
        input_ = Concatenate(axis=-1)([sketch_input, real_input])
        e1 = encoder_block(input_, unit=f, kernel=4, strides=2, padding='same', bn=False) #128
        e2 = encoder_block(e1, unit=f*2, kernel=4, strides=2, padding='same') #64
        e3 = encoder_block(e2, unit=f*4, kernel=4, strides=2, padding='same') #32
        e4 = encoder_block(e3, unit=f*8, kernel=4, strides=2, padding='same') #16
        e5 = encoder_block(e4, unit=f*8, kernel=4, strides=2, padding='same') #8
        e6 = encoder_block(e5, unit=f*8, kernel=4, strides=2, padding='same') #4
        e7 = encoder_block(e6, unit=f*8, kernel=4, strides=2, padding='same') #2
        out = Conv2D(f*16, kernel_size=4, strides=2, padding='same')(e7)
        # out = Flatten()(out) #512
        mean_out = Dense(1024)(out)
        log_var_out = Dense(1024)(out)
        model = Model(inputs=[sketch_input, real_input], outputs=[mean_out, log_var_out])
        model.summary()
        return model

    def build_Decoder(self):
        def encoder_block(input_, unit, kernel=4, strides=2, padding='same', bn=True):
            x = Conv2D(unit, kernel_size=kernel, strides=strides, padding=padding)(input_)
            if bn:
                x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x
        def decoder_block(input_, unit, kernel=4, strides=2, padding='same', bn=True):
            x = Conv2DTranspose(unit, kernel_size=kernel, strides=strides, padding=padding)(input_)
            if bn:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x

        f = self.config.d_first_unit
        f_e = self.config.e_first_unit
        sketch_input = Input(shape=(256, 256, 3))
        e1 = encoder_block(sketch_input, unit=f_e, kernel=4, strides=2, padding='same', bn=False)  # 128
        e2 = encoder_block(e1, unit=f_e * 2, kernel=4, strides=2, padding='same')  # 64
        e3 = encoder_block(e2, unit=f_e * 4, kernel=4, strides=2, padding='same')  # 32
        e4 = encoder_block(e3, unit=f_e * 8, kernel=4, strides=2, padding='same')  # 16
        e5 = encoder_block(e4, unit=f_e * 8, kernel=4, strides=2, padding='same')  # 8
        e6 = encoder_block(e5, unit=f_e * 8, kernel=4, strides=2, padding='same')  # 4
        e7 = encoder_block(e6, unit=f_e * 8, kernel=4, strides=2, padding='same')  # 2
        s = Conv2D(f_e * 16, kernel_size=4, strides=2, padding='same')(e7)

        latent_input = Input(shape=(1, 1, 1024))
        input_ = Concatenate(axis=-1)([s, latent_input])
        d1 = decoder_block(input_, unit=f, kernel=4, strides=2, padding='same', bn=False)
        d2 = decoder_block(d1, unit=f, kernel=4, strides=2, padding='same')
        d3 = decoder_block(d2, unit=f, kernel=4, strides=2, padding='same')
        d4 = decoder_block(d3, unit=f, kernel=4, strides=2, padding='same')
        d5 = decoder_block(d4, unit=int(f / 2), kernel=4, strides=2, padding='same')
        d6 = decoder_block(d5, unit=int(f / 4), kernel=4, strides=2, padding='same')
        d7 = decoder_block(d6, unit=int(f / 8), kernel=4, strides=2, padding='same')
        out = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(d7)

        model = Model(inputs=[sketch_input, latent_input], outputs=out)
        # plot_model(model=model,show_shapes=True,to_file='./decoder.png')
        model.summary()
        return model

    def build_CVAE(self):
        def sampling(z_mean, z_log_var):
            batch = K.shape(z_mean)[0]
            # dim = K.shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, 1, 1, 1024))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        def loss(real, fake, z_mean, z_log_var):
            reconstruction_loss = mean_squared_error(real, fake)
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            cvae_loss = K.mean(256*reconstruction_loss + kl_loss)
            return cvae_loss
        sketch_input = Input(shape=(256,256,3))
        real_input = Input(shape=(256,256,3))

        z_mean, z_log_var = self.Encoder([sketch_input, real_input])
        latent = sampling(z_mean, z_log_var)
        out = self.Decoder([sketch_input, latent])
        model = Model(inputs=[sketch_input, real_input], outputs=out)
        model.add_loss(losses=loss(real_input, out, z_mean, z_log_var))
        model.compile(optimizer=Adam(self.config.lr, 0.5))
        return model

    def train(self):
        s = time.time()
        history = self.CVAEmodel.fit(x=[self.sketchdata, self.realdata], y=self.realdata,
                                     batch_size=self.config.batchsize, epochs=self.config.epochs,
                                     callbacks=[self.save_image_callback])
        self.loss = history.history['loss']
        # plt.show()
        e = time.time()
        self.finish_training(used_time=round(e-s, 4))
        self.predict()

    def finish_training(self, used_time):
        self.path.make_folder()
        self.config.save_to_json(used_time=used_time, path=f'{self.path.CONFIGPATH}/config.json')
        plot_model(self.Encoder, to_file=f'{self.path.MODELPATH}/{self.modelname}_Encoder.png', show_shapes=True,
                   show_layer_names=True)
        plot_model(self.Decoder, to_file=f'{self.path.MODELPATH}/{self.modelname}_Decoder.png', show_shapes=True,
                   show_layer_names=True)
        self.Decoder.save(f'{self.path.MODELPATH}/{self.modelname}_decoder.h5')  # 生成器存特定資料夾，其他隨便存，反正不是重點
        self.CVAEmodel.save(f'{self.path.H5PATH}/{self.modelname}.h5')


        np.save(f'{self.path.LOSSPATH}/loss.npy', arr=self.loss)

        loss = np.load(f'{self.path.LOSSPATH}/loss.npy', allow_pickle=True)

        plt.clf()
        plt.title('Loss')
        plt.plot(loss)
        plt.legend(loc='best')
        plt.savefig(f"{self.path.LOSSPATH}/VAE-Loss.png")

        video.transform_video(2, path=f'{self.path.TRAININGIMGPATH}/*.png',
                              result_name=f'{self.path.PROCESSPATH}/training_process.mp4')
    def predict(self):
        self.test_sketchdata, self.test_realdata = DataLoader().load_testing_data()
        self.idx = np.array([10, 20, 30])
        self.test_sketchdata = self.test_sketchdata[self.idx]
        self.test_realdata = self.test_realdata[self.idx]
        fake_images = self.CVAEmodel.predict([self.test_sketchdata, self.test_realdata])
        fake_images = ((fake_images + 1) / 2)

        noise = np.random.random(size=(3, 1, 1, 1024))
        d_fake_images = self.Decoder.predict([self.test_sketchdata, noise])
        d_fake_images = ((d_fake_images + 1) / 2)

        plt.figure(figsize=(8, 8))
        plt.title(f'The Images Of {self.modelname} Generate', fontsize=17)
        for j in range(3):
            plt.subplot(3, 4, 4 * j + 1)
            plt.title('sketch', fontsize=12)
            plt.imshow(self.test_sketchdata[j])
            plt.axis("off")

            plt.subplot(3, 4, 4 * j + 2)
            plt.title('CVAE generated', fontsize=12)
            plt.imshow(fake_images[j])
            plt.axis("off")

            plt.subplot(3, 4, 4 * j + 3)
            plt.title('Decoder generated', fontsize=12)
            plt.imshow(d_fake_images[j])
            plt.axis("off")

            plt.subplot(3, 4, 4 * j + 4)
            plt.title('true', fontsize=12)
            plt.imshow(self.test_realdata[j])
            plt.axis("off")
        plt.savefig(f'{self.path.PROCESSPATH}/predict.png')

# if __name__=='__main__':
#     from utils import config
#
#     config = config.CVAE_TrainConfig(epochs=80000,  # 80000
#                                 batchsize=1,
#                                 lr=0.0002,
#                                 e_first_unit=64,
#                                 d_first_unit=512)
#     vae = CVAE(model_name='CVAE', config=config)
#     vae.train()