import tensorflow as tf
import numpy as np
from scipy import linalg
from tensorflow.keras.metrics import Metric,Mean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Lambda,GlobalAveragePooling2D
from keras.layers import Rescaling,Resizing
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from metrics import get_LPIPS_distance


def fid_get_features(images): #計算圖片特徵
    model = InceptionV3(weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
    # 預處理圖片，將其轉換為Inception v3模型的輸入格式
    # images = Resizing(height=256, width=256)(images)
    images = preprocess_input(images)
    features = model.predict(images)
    return features
def calculate_FID(features_x, features_g):
    # 根據公式，x代表真實圖片，g代表生成圖片
    # 計算真實圖片和生成圖片之特徵向量的平均值和共變異數矩陣
    # 平均值
    mu_x = np.mean(features_x, axis=0)
    mu_g = np.mean(features_g, axis=0)
    # 共變異數矩陣
    sigma_x = np.cov(features_x, rowvar=False) #shape=(2048, 2048)
    sigma_g = np.cov(features_g, rowvar=False)
    # 平均值之間的平方距離(square distance)
    square_distance = np.sum((mu_x - mu_g) ** 2)
    # 共變異數矩陣的幾何平均
    covariance_mean = linalg.sqrtm(sigma_x.dot(sigma_g))
    # 有些元素可能平方過後變成複數，所以要把複數都取實部
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    fid = square_distance + np.trace(sigma_x + sigma_g - 2*covariance_mean)
    return fid



class KID(Metric):
    def __init__(self, name, image_size, kid_image_size, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = Sequential(
            [
                Input(shape=(image_size, image_size, 3)),#64
                Rescaling(255.0),
                Resizing(height=kid_image_size, width=kid_image_size),#75
                Lambda(preprocess_input),
                InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()



if __name__=='__main__':
    img_real = np.load(r"C:\Users\陳冠霖\Desktop\Cycle_GAN-4\Cycle_GAN\real_images.npy")
    img_fake = np.load(r"C:\Users\陳冠霖\Desktop\Cycle_GAN-4\Cycle_GAN\fake_images.npy")
    print(img_real.shape)
    print(img_fake.shape)
    #
    psnr = tf.image.psnr(img_real.copy(), img_fake.copy(), max_val=1)
    print('PSNR between img_real and img_fake is:', np.average(psnr))

    ssim = tf.image.ssim(img_real.copy(), img_fake.copy(), max_val=1)
    print('SSIM between img_real and img_fake is:', np.average(ssim))

    kid = KID(name='kid', image_size=256, kid_image_size=256)
    kid.update_state(img_real.copy(), img_fake.copy(), sample_weight=None)
    print('KID between img_real and img_fake is:', kid.result())
    #
    features_x = fid_get_features(img_real.copy())  # 圖片的shape為 (n, 圖片寬, 圖片高, 3)
    features_g = fid_get_features(img_fake.copy())
    fid = calculate_FID(features_x, features_g)
    print('FID between img_real and img_fake is:', fid)

    # img_real = (img_real.copy()*2)-1
    # img_fake = (img_fake.copy()*2)-1
    img_real = img_real.reshape((-1,3,256,256))
    img_fake = img_fake.reshape((-1,3,256,256))
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    import torch
    img_real = torch.from_numpy(img_real)
    img_fake = torch.from_numpy(img_fake)
    # img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    # img1 = torch.zeros(1, 3, 64, 64)
    d = loss_fn_alex.forward(img_real, img_fake)
    print(d.shape)
    print(torch.sum(d)/d.shape[0])