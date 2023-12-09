import numpy as np
import keras
from dataset import DataLoader
from tensorflow.keras.layers import Input,Dense,LeakyReLU,Activation,\
            Conv2D,Flatten,Reshape,BatchNormalization,Embedding,multiply,Dropout,MaxPooling2D
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import time

class CGAN():

    def __init__(self):

        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 256
        self.depth = 64
        self.p = 0.4
        self.dloss_lst = []
        self.gloss_lst = []
        d = DataLoader()
        self.sketchdata, self.realdata = d.load_data(metrics='[0,1]')


        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label

        noise = Input(shape=(256,256,3))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img,noise])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(inputs=noise, outputs=valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    z_dimensions = 3
    def build_generator(self, p=0.45):

        image1 = Input((256,256,3))


        # 第 1 密集層
        dense1 = Dense(3)(image1)
        dense1 = BatchNormalization(momentum=0.9)(dense1)  # default momentum for moving average is 0.99
        dense1 = Activation(activation='relu')(dense1)
        dense1 = keras.layers.Reshape((256,256,3))(dense1)
        dense1 = Dropout(p)(dense1)


        # 反卷積層
        conv1 = UpSampling2D()(dense1)
        conv1 = Conv2DTranspose(3,
                                kernel_size=3, padding='same',
                                activation=None, )(conv1)
        conv1 = BatchNormalization(momentum=0.9)(conv1)
        conv1 = Activation(activation='relu')(conv1)

        conv2 = UpSampling2D()(conv1)
        conv2 = Conv2DTranspose(2,
                                kernel_size=3, padding='same',
                                activation=None, )(conv2)
        conv2 = BatchNormalization(momentum=0.9)(conv2)
        conv2 = Activation(activation='relu')(conv2)

        conv3 = Conv2DTranspose(1,
                                kernel_size=3, padding='same',
                                activation=None, )(conv2)
        conv3 = BatchNormalization(momentum=0.9)(conv3)
        conv3 = Activation(activation='relu')(conv3)

        # 輸出層
        image = Conv2D(3, kernel_size=5, strides=(4, 4),padding='same',
                       activation='sigmoid')(conv3)

        model = Model(inputs=image1, outputs=image)

        model.summary()

        return model

    def build_discriminator(self,depth=64, p=0.4):

        image = Input((256,256,3))
        sketch_image = Input((256,256,3))

        concat = keras.layers.Concatenate(axis=-1)([image, sketch_image])

        # 卷積層
        conv1 = Conv2D(depth * 1, 5, strides=2,
                       padding='same', activation='relu')(concat)
        conv1 = Dropout(p)(conv1)

        conv2 = Conv2D(depth * 2, 5, strides=2,
                       padding='same', activation='relu')(conv1)
        conv2 = Dropout(p)(conv2)

        conv3 = Conv2D(depth * 4, 5, strides=2,
                       padding='same', activation='relu')(conv2)
        conv3 = Dropout(p)(conv3)

        conv4 = Conv2D(depth * 8, 5, strides=1,
                       padding='same', activation='relu')(conv3)
        conv4 = Flatten()(Dropout(p)(conv4))

        # 輸出層
        prediction = Dense(1, activation='sigmoid')(conv4)

        # 定義模型
        model = Model(inputs=[image,sketch_image], outputs=prediction)

        model.summary()

        return model



    def train(self, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, self.sketchdata.shape[0], batch_size)
            imgs, labels = self.sketchdata[idx], self.realdata[idx]
            gen_imgs = self.generator.predict(imgs)


            d_loss_real1 = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake1 = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss_real2 = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake2 = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss1 = 0.5 * np.add(d_loss_real1, d_loss_fake1)
            d_loss2 = 0.5 * np.add(d_loss_real2, d_loss_fake2)
            d_loss = 0.5 * (d_loss1 + d_loss2)


            g_loss1 = self.combined.train_on_batch([imgs, ], valid)
            g_loss2 = self.combined.train_on_batch([imgs, ], valid)
            g_loss3 = self.combined.train_on_batch([imgs, ], valid)

            g_loss = 0.33 * (g_loss1 + g_loss2 + g_loss3)

            print ("epoch: %d, [Discriminator loss: %f] [Generator loss: %f]" % (epoch, d_loss[0], g_loss))
            self.dloss_lst.append(d_loss[0])
            self.gloss_lst.append(g_loss)

            if epoch < 1001:
                if epoch % 50 == 0:
                    self.predict(epoch=epoch)

            elif epoch % 500 == 0:
                    self.predict(epoch=epoch)

        # self.combined.save('./model/combind.h5')
        self.generator.save('./CGAN_images/model/generator.h5')
        # self.discriminator.save('./model/discriminator.h5')

        # np.save(file='./dloss.npy',arr=np.array(self.dloss_lst))
        # np.save(file='./gloss.npy', arr=np.array(self.gloss_lst))

    def predict(self, num_images=3, training=True, model=None, **kwargs):
        if training:
            idx = np.array([10, 20, 30])
            fake_images = self.generator.predict(self.sketchdata[idx])
            # fake_images = ((fake_images + 1) / 2)
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
                plt.imshow(self.realdata[idx][j])
                plt.axis("off")
            plt.savefig(f'./CGAN_images/epoch{kwargs["epoch"] + 1:0>5}.png')
        else:
            test_sketchdata, test_realdata = DataLoader().load_testing_data(metrics='[0,1]')
            idx = np.array([10, 20, 30])
            if model==None:
                generated_images = self.generator.predict(test_sketchdata[idx])
            else:
                model = load_model(model)
                generated_images = model.predict(test_sketchdata[idx])
            # generated_images = generated_images  # .astype(np.uint8)
            plt.figure(figsize=(8, 8))
            plt.title(f'predict')
            plt.suptitle(f'The Images of CGAN Generate', fontsize=17)
            for j in range(num_images):
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
            plt.savefig(f'./CGAN_images/predict.png')


if __name__ == '__main__':
    cgan = CGAN()
    start = time.time()
    cgan.train(epochs=30000, batch_size=64, sample_interval=200)
    end = time.time()
    print(end-start)
    cgan.predict(training=False)
