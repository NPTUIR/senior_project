import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import facade
import requests
import datetime
from tensorflow.keras.utils import plot_model
import tensorflowjs as tfjs
from final_predict import save_fig ,get_transform_data,predict

def create_VAE(latent_dim):
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    encoder_out = tf.keras.layers.Dense(latent_dim + latent_dim)(x)

    # mean, logvar = tf.split(encoder_out, num_or_size_splits=2, axis=-1)
    # x = tf.keras.layers.Lambda(custom_layer)([mean, logvar])

    Decoder_input = tf.keras.layers.Input((latent_dim, ))
    x = tf.keras.layers.Dense(units=16 * 16 * 128, activation='relu')(Decoder_input)
    x = tf.keras.layers.Reshape(target_shape=(16, 16, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')(x)
    out = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=(1, 1), padding="SAME", activation='sigmoid')(x)

    encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_out)
    decoder_model = tf.keras.Model(inputs=Decoder_input, outputs=out)

    input_ = tf.keras.layers.Input(shape=(256, 256, 3))
    mean, logvar = tf.split(encoder_model(input_), num_or_size_splits=2, axis=-1)
    x = tf.keras.layers.Lambda(custom_layer)([mean, logvar])
    out = decoder_model(x)


    model = tf.keras.Model(inputs=input_, outputs=[out, mean, logvar])
    model.summary()
    return model, encoder_model, decoder_model

def custom_layer(input_):
    mean, logvar = input_
    eps = tf.random.normal(shape=tf.shape(mean))
    z = eps * tf.exp(logvar * 0.5) + mean
    return z


Allkl_loss = []
Allrecon_loss =[]
# 損失函數
def compute_loss(model, x, real_data):#x=sketch_data->input
    x_recon, mean, logvar = model(x)
    #recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_data, x_recon))
    recon_loss = tf.reduce_sum(tf.square(real_data - x_recon), axis=[1, 2, 3])
    Allrecon_loss.append(tf.reduce_mean(recon_loss))
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=-1)
    Allkl_loss.append(tf.reduce_mean(kl_loss))
    loss = tf.reduce_mean(recon_loss + kl_loss)
    return loss

# 梯度下降優化器
def train_step(model, x, real_data, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, real_data)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#generate
def generate_data(model, n_samples, input_):
    # z = tf.random.normal(shape=(n_samples, latent_dim))
    idx = np.random.randint(0, input_.shape[0], n_samples)
    x = input_[idx]
    x_generated, _, _ = model(x)
    # mean, logvar = encoder(x)
    # eps = tf.random.normal(shape=mean.shape)
    # z = eps * tf.exp(logvar * 0.5) + mean
    # x_generated = decoder(z)
    return x_generated



#LineNotify
class Code_returner:
    def __init__(self,token=''):
        assert token!='','no token'
        self.headers = {
                        "Authorization": "Bearer "+token,
                        #"Content-Type":"application/x-www-form-urlencoded"
                        }
    def send_message(self,message:str):
        params = {"message": f'[INFO] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:\n {message}'}
        requests.post('https://notify-api.line.me/api/notify', headers=self.headers, params=params)
    def send_image(self,img_path:str,message=None):
        img=open(img_path,'rb')
        files = {"imageFile": img}
        data = {'message': f'[INFO] {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:\n {message}'}
        requests.post('https://notify-api.line.me/api/notify', headers=self.headers, files=files, data=data)



if __name__=='__main__':
    # 可改參數---------------
    num_epochs = 9000
    batch_size = 64
    latent_dim = 1024 #16

    # 可改參數---------------
    elf = Code_returner(
        token='lT8ZnYtG6gVYlMWttzydJwKzZclGxJbdwWfLzLdw8ns')  # lT8ZnYtG6gVYlMWttzydJwKzZclGxJbdwWfLzLdw8ns
    # 載入資料集
    d = facade.DataLoader()
    sketchdata, realdata = d.load_data(normalization='[0,1]',data_argumantation = True)
    test_sketchdata, test_realdata = d.load_testing_data()
    num_batches = sketchdata.shape[0] // batch_size
    # 建立VAE
    #    return model, encoder_model, decoder_model

    VAE_model, encoder_model, decoder_model = create_VAE(latent_dim)
    plot_model(model=VAE_model, to_file='data/model.png', show_shapes=True)
    # decoder_model = create_decoder(latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # 訓練VAE
    AllLoss=[]
    loss_lst = np.zeros(shape=(num_batches))
    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            x_batch = sketchdata[start_idx:end_idx]
            real_batch = realdata[start_idx:end_idx]
            loss = train_step(VAE_model, x_batch, real_batch, optimizer)
            loss_lst[batch_idx] = loss
        print("Epoch: {}, Loss: {}".format(epoch+1, loss))
        AllLoss.append(loss)
        if epoch%40 == 0:
            # predict
            # try:
            predict(VAE_model,sketchdata,realdata, epoch=epoch)
            # except Exception as e:
            #     print(epoch)
            #     print(e)

    test_sketchdata, test_realdata = d.load_testing_data()
    idx = np.array([10, 20, 30])
    generated_images, _, _ = VAE_model.predict(test_sketchdata[idx])
    # generated_images = ((generated_images + 1) / 2) #反正規化。我用tanh所以要這樣做，用sigmoid輸出則不用反正規化
    plt.figure(figsize=(8, 8))
    plt.title(f'predict')
    plt.suptitle(f'The Images of CVAE Generate', fontsize=17)  # {self.modelname}可以改成你的模型名稱例如CVAE、CGAN、Cycle-GAN
    num_images=3
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
    plt.savefig(f'./images/predict.png')  # 儲存圖片
    # try:
    # save_fig(VAE_model)
    # get_transform_data(VAE_model)

    # except Exception as e:
    #     print(e)
    #     elf.send_message(message='fail to save png')
    #     elf.send_message(message= e)



    np.save('data/loss.npy', np.array(AllLoss))
    np.save('data/KLloss.npy', np.array(Allkl_loss))
    np.save('data/reconloss.npy', np.array(Allrecon_loss))

    try:
        #tfjs.converters.save_keras_model(model=encoder_model, artifacts_dir=f'./tfjsmodel/encoder')
        #tfjs.converters.save_keras_model(model=decoder_model, artifacts_dir=f'./tfjsmodel/decoder')
        VAE_model.save('VAE_model.h5')
        # decoder_model.save('decoder_model.h5')
    except Exception as e:
        print(e)
        elf.send_message(message='fail to save model')


    elf.send_message(message='Finish')
    elf.send_image(img_path='./images/predict.png')
    print('Connect to Line successfully')