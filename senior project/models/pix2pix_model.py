import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LeakyReLU, Activation, Conv2D, Conv2DTranspose, Dropout, Concatenate, ZeroPadding2D
from tensorflow_addons.layers import InstanceNormalization
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

class Pix2pix:
    def __init__(self, model_name=None, config=None):
        if model_name == None:
            self.modelname = self.__class__.__name__
        else:
            self.modelname = model_name
        self.path = Path(model_name=model_name)
        self.config = config
        # self.binary_crossentropy_loss = BinaryCrossentropy(from_logits=True)

        self.genModel = self.build_generator()
        self.disModel = self.build_discriminator()
        self.advModel = self.build_adversialmodel()
        self.disc_patch = (30, 30, 1)
        self.sketchdata, self.realdata =DataLoader().load_data(normalization='[-1,1]')

        test_sketchdata, test_realdata = DataLoader().load_testing_data()

        np.save(file='test_sketch.npy',arr=test_sketchdata)
        np.save(file='test_real.npy', arr=test_realdata)

    def encoder_block(self, input_, u, k, s=2, p='same', bn=True, act='leakyrelu'):
        x = Conv2D(u, kernel_size=k, strides=s, padding=p, use_bias=False, kernel_initializer=RandomNormal(mean=0.0,stddev=0.02), bias_initializer=Zeros())(input_)
        if bn:
            x = InstanceNormalization()(x)
        if act == 'leakyrelu':
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def decoder_block(self, input_, u, k, s=2, p='same', bn=True, dropout=True, act='relu'):
        x = Conv2DTranspose(u, kernel_size=k, strides=s, padding=p, use_bias=False, kernel_initializer=RandomNormal(mean=0.0,stddev=0.02), bias_initializer=Zeros())(input_)
        if bn:
            x = InstanceNormalization()(x)
        if dropout:
            x = Dropout(0.5)(x)
        if act == 'relu':
            x = Activation('relu')(x)
        elif act == 'leakyrelu':
            x = LeakyReLU(alpha=0.2)(x)
        elif act == 'sigmoid':
            x = Activation('sigmoid')(x)
        elif act == 'tanh':
            x = Activation('tanh')(x)
        return x

    # @property
    def build_generator(self):
        input_layer = Input(shape=(256, 256, 3))
        f = self.config.g_first_unit
        en1 = self.encoder_block(input_layer, u=int(f/2), k=4, bn=False)
        en2 = self.encoder_block(en1, u=f, k=4)
        en3 = self.encoder_block(en2, u=f*2, k=4)
        en4 = self.encoder_block(en3, u=f*4, k=4)
        en5 = self.encoder_block(en4, u=f*4, k=4) #8
        en6 = self.encoder_block(en5, u=f*4, k=4)
        en7 = self.encoder_block(en6, u=f*4, k=4)
        en8 = self.encoder_block(en7, u=f*4, k=4) #2
        latent = self.encoder_block(en8, u=f*4, k=4) #1
        de0 = self.decoder_block(latent, u=f*4, k=4, s=1)
        de0 = Concatenate(axis=-1)([de0, en8])
        de1 = self.decoder_block(de0, u=f*8, k=4)
        de1 = Concatenate(axis=-1)([de1, en7])
        de2 = self.decoder_block(de1, u=f*8, k=4)
        de2 = Concatenate(axis=-1)([de2, en6])
        de3 = self.decoder_block(de2, u=f*8, k=4, dropout=False)
        de3 = Concatenate(axis=-1)([de3, en5])
        de4 = self.decoder_block(de3, u=f*8, k=4, dropout=False)
        de4 = Concatenate(axis=-1)([de4, en4])
        de5 = self.decoder_block(de4, u=f*4, k=4, dropout=False)
        de5 = Concatenate(axis=-1)([de5, en3])
        de6 = self.decoder_block(de5, u=f*2, k=4, dropout=False)
        de6 = Concatenate(axis=-1)([de6, en2])
        de7 = self.decoder_block(de6, u=f, k=4, dropout=False)
        de7 = Concatenate(axis=-1)([de7, en1])
        out = self.decoder_block(de7, u=3, k=4, dropout=False, bn=False, act='tanh')

        model = Model(inputs=input_layer, outputs=out)
        # model.summary()
        return model

    def build_discriminator(self):
        sketch_input_layer = Input(shape=(256, 256, 3))
        img_input_layer = Input(shape=(256, 256, 3))
        input_layer = Concatenate(axis=-1)([img_input_layer, sketch_input_layer])
        f = self.config.d_first_unit
        x = self.encoder_block(input_layer, u=f, k=4, bn=False)
        x = self.encoder_block(x, u=f*2, k=4)
        x = self.encoder_block(x, u=f*4, k=4)
        x = ZeroPadding2D()(x)
        x = self.encoder_block(x, u=f*8, k=4, s=1, p='valid')
        x = ZeroPadding2D()(x)
        out = Conv2D(1, kernel_size=4, strides=1, padding='valid', kernel_initializer=RandomNormal(mean=0.0,stddev=0.02), bias_initializer=Zeros())(x)

        model = Model(inputs=[img_input_layer, sketch_input_layer],outputs=out)
        # model.summary()
        model.compile(loss='mse', optimizer=Adam(self.config.dis_lr, 0.5), metrics=['accuracy'])
        return model

    def build_adversialmodel(self):
        sketch_input_layer = Input(shape=(256, 256, 3),name='sketch')
        sketch2real = self.genModel(sketch_input_layer)
        self.disModel.trainable = False
        valid = self.disModel([sketch2real, sketch_input_layer])
        model = Model(inputs=sketch_input_layer, outputs=[valid, sketch2real])
        model.compile(loss=['mse','mae'], loss_weights=[1, self.config.lambda_], optimizer=Adam(self.config.gen_lr, 0.5))

        return model

    def train(self):
        self.dloss = []
        self.gloss = []
        s = time.time()
        for i in range(self.config.epochs):
            idx = np.random.randint(0, self.sketchdata.shape[0], self.config.batchsize)
            fake_images = self.genModel(self.sketchdata[idx])
            real_images = self.realdata[idx]
            real_sketch = self.sketchdata[idx]

            y_real = np.ones((self.config.batchsize, ) + self.disc_patch)
            y_fake = np.zeros((self.config.batchsize, ) + self.disc_patch)

            d_loss_real = self.disModel.train_on_batch([real_images, real_sketch], y_real)
            d_loss_fake = self.disModel.train_on_batch([fake_images, real_sketch], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            self.dloss.append(d_loss[0])

            g_loss1 = self.advModel.train_on_batch(real_sketch, [y_real, real_images])
            g_loss2 = self.advModel.train_on_batch(real_sketch, [y_real, real_images])
            g_loss = 0.5 * np.add(g_loss1, g_loss2)
            self.gloss.append(g_loss)
            print(f"epochs:{i + 1} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.3f}] [G loss: {g_loss}]")
            if (i + 1) % 500 == 0 or i == 0 or (i+1 <= 3000 and (i+1)%40==0):
                self.predict(epoch=i)
        e = time.time()
        self.finish_training(used_time=round(e-s,4))
    def finish_training(self,used_time):
        self.path.make_folder()
        self.config.save_to_json(used_time=used_time, path=f'{self.path.CONFIGPATH}/config.json')
        plot_model(self.genModel, to_file=f'{self.path.MODELPATH}/{self.modelname}_generator.png', show_shapes=True, show_layer_names=True)
        plot_model(self.disModel, to_file=f'{self.path.MODELPATH}/{self.modelname}_discriminator.png', show_shapes=True, show_layer_names=True)
        plot_model(self.advModel, to_file=f'{self.path.MODELPATH}/{self.modelname}_adversial.png', show_shapes=True, show_layer_names=True)
        self.genModel.save(f'{self.path.MODELPATH}/{self.modelname}_generator.h5') #生成器存特定資料夾，其他隨便存，反正不是重點
        self.disModel.save(f'{self.path.H5PATH}/{self.modelname}_discriminator.h5')
        self.advModel.save(f'{self.path.H5PATH}/{self.modelname}_adversarial.h5')

        np.save(f'{self.path.LOSSPATH}/gloss.npy', arr=self.gloss)
        np.save(f'{self.path.LOSSPATH}/dloss.npy', arr=self.dloss)

        gloss = np.load(f'{self.path.LOSSPATH}/gloss.npy', allow_pickle=True)
        dloss = np.load(f'{self.path.LOSSPATH}/dloss.npy', allow_pickle=True)
        plt.clf()
        plt.title('Loss')
        plt.plot(gloss[:,1], label='g')
        plt.plot(dloss, label='d')
        plt.legend(loc='best')
        plt.savefig(f"{self.path.LOSSPATH}/Loss.png")

        plt.clf()
        plt.title('Generated MAE Loss')
        plt.plot(gloss[:,2], label='g')
        plt.legend(loc='best')
        plt.savefig(f"{self.path.LOSSPATH}/Generated MAE Loss.png")
        video.transform_video(2, path=f'{self.path.TRAININGIMGPATH}/*.png', result_name=f'{self.path.PROCESSPATH}/training_process.mp4')

    def predict(self, num_images=3, training=True, **kwargs):
        if training:
            idx = np.array([10, 20, 30])
            fake_images = self.genModel.predict(self.sketchdata[idx])
            fake_images = ((fake_images + 1) / 2)
            plt.figure(figsize=(8, 8))
            plt.title(f'Training Process of {kwargs["epoch"] + 1} Epochs')
            for j in range(num_images):
                plt.subplot(num_images, num_images, num_images * j + 1)
                plt.title('sketch', fontsize=12)
                plt.imshow(self.sketchdata[idx][j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j + 2)
                plt.title('generated', fontsize=12)
                plt.imshow(fake_images[j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j + 3)
                plt.title('true', fontsize=12)
                plt.imshow((self.realdata[idx][j] + 1) / 2)
                plt.axis("off")
            plt.savefig(f'{self.path.TRAININGIMGPATH}/epoch{kwargs["epoch"] + 1:0>5}.png')
        else:
            test_sketchdata, test_realdata = DataLoader().load_testing_data()
            idx = np.array([10, 20, 30])
            model = load_model(f'{self.path.MODELPATH}/{self.modelname}_generator.h5')
            generated_images = model.predict(test_sketchdata[idx])
            generated_images = ((generated_images + 1) / 2)#.astype(np.uint8)
            plt.figure(figsize=(8, 8))
            plt.title(f'predict')
            plt.suptitle(f'The Images Of {self.modelname} Generate', fontsize=17)
            for j in range(num_images):
                plt.subplot(num_images, num_images, num_images * j+1)
                plt.title('sketch', fontsize=12)
                plt.imshow(test_sketchdata[j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images*j+2)
                plt.title('generated', fontsize=12)
                plt.imshow(generated_images[j])
                plt.axis("off")

                plt.subplot(num_images, num_images, num_images * j+3)
                plt.title('true', fontsize=12)
                plt.imshow(test_realdata[j])
                plt.axis("off")
            plt.savefig(f'{self.path.PROCESSPATH}/predict.png')



# if __name__ == '__main__':
    # gan = Pix2pix(model_name='pix2pix')
    # s = time.time()
    # gan.train() # 10000 epochs->used 25826.565 seconds7.17hour
    # e = time.time()
    # print(f'used {e-s:.3f} seconds')
    #
    # gan.predict()

