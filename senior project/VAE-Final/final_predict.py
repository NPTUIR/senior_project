import numpy as np
import matplotlib.pyplot as plt
import facade

def predict(model,sketchdata,realdata, num_images=3, training=True, **kwargs):
    d = facade.DataLoader()
    if training:
        #sketchdata, realdata = d.load_data(normalization='[0,1]')
        idx = np.array([10, 20, 30]) #選擇第11、21、31張圖片，雖然沒有意義但統一比較好~
        fake_images, _, _ = model.predict(sketchdata[idx]) #生成器生成圖片
        #fake_images = ((fake_images + 1) / 2) #反正規化。我用tanh所以要這樣做，用sigmoid輸出則不用反正規化
        plt.figure(figsize=(8, 8))
        plt.title(f'Training Process of {kwargs["epoch"] + 1} Epochs')
        for j in range(num_images):
            plt.subplot(num_images, num_images, num_images * j + 1)
            plt.title('Input Sketch Image', fontsize=12)
            plt.imshow(sketchdata[idx][j])
            plt.axis("off")
            # print(len(fake_images))
            # print(fake_images[1].shape)
            plt.subplot(num_images, num_images, num_images * j + 2)
            plt.title('Generated Image', fontsize=12)
            plt.imshow(fake_images[j])
            plt.axis("off")

            plt.subplot(num_images, num_images, num_images * j + 3)
            plt.title('Ground Truth', fontsize=12)
            plt.imshow(realdata[idx][j])
            plt.axis("off")
        plt.savefig(f'./images/epoch{kwargs["epoch"] + 1:0>5}.png') #儲存圖片 kwargs["epoch"] + 1:0>5這邊不要改，否則保證之後影片生成時你會哭
    else:
        test_sketchdata, test_realdata = d.load_testing_data()
        idx = np.array([10, 20, 30])
        generated_images, _, _ = model.predict(test_sketchdata[idx])
        #generated_images = ((generated_images + 1) / 2) #反正規化。我用tanh所以要這樣做，用sigmoid輸出則不用反正規化
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        plt.suptitle(f'The Images of CVAE Generate', fontsize=17) #{self.modelname}可以改成你的模型名稱例如CVAE、CGAN、Cycle-GAN
        for j in range(num_images): #這邊都不要動
            plt.subplot(num_images, num_images, num_images * j+1)
            plt.title('Input Sketch Image', fontsize=12)
            plt.imshow(test_sketchdata[j])
            plt.axis("off")

            plt.subplot(num_images, num_images, num_images*j+2)
            plt.title('Generated Image', fontsize=12)
            plt.imshow(generated_images[j])
            plt.axis("off")

            plt.subplot(num_images, num_images, num_images * j+3)
            plt.title('Ground Truth', fontsize=12)
            plt.imshow(test_realdata[j])
            plt.axis("off")
        plt.savefig(f'./images/predict.png') #儲存圖片




def get_transform_data(model,
                       model_name='CVAE',
                       dir=r'C:\Users\user\PycharmProjects\pythonProject\Hsieh\VAE-Final\sketchdata\*.jpg'):
    import glob, os, cv2
    import matplotlib.pyplot as plt
    import numpy as np
    # 路徑要改
    dir = glob.glob(dir)
    # h5模型要改一下
    dataname = [f'1{i}_B' for i in range(10)] + ['1_B'] + [f'2{i}_B' for i in range(10)] + ['2_B'] + ['30_B', '31_B', '32_B'] + [f'{i}_B' for i in range(3, 10)]
    for i in range(32):
        datapath = dir[i]
        data = cv2.imread(datapath, cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
        data = data.astype(np.float32) / 255

        pred, _, _ = model.predict(data)
        # pred = (pred + 1) / 2
        plt.imsave(f'web result/{model_name}/{dataname[i]}.jpg', pred[0])
# 最後儲存模型生成32張圖片的部分
def save_fig(model):
    """

    Args:
        self:
        model: 模型的h5路徑
        這邊會用16張training data與16張testing data一共32張圖片。

    Returns: 16張訓練資料假圖+16張測試資料假圖共32張。16張訓練資料真實圖+16張測試資料真實圖共32張

    """

    # 用訓練資料生成真實圖
    d = facade.DataLoader()
    sketchdata, realdata = d.load_data(normalization='[0,1]', data_argumantation=True)
    #
    idx = np.array([i * 5 for i in range(16)])
    fake_images, _, _ = model.predict(sketchdata[idx])  # 生成假圖片
    real_images = realdata[idx]  # 對真實圖片反正規化。我用tanh所以要這樣做，用sigmoid輸出則不用反正規化

    # 用測試資料生成真實圖
    test_sketchdata, test_realdata = d.load_testing_data()
    test_fake_images, _, _ = model.predict(test_sketchdata[idx])
    test_real_images = test_realdata[idx]  # 真實圖片
    fake_images = np.append(fake_images, test_fake_images, axis=0).astype(np.float32)
    real_images = np.append(real_images, test_real_images, axis=0).astype(np.float32)
    np.save(arr=fake_images, file=f'./result/VAE/fake_images.npy')
    np.save(arr=real_images, file=f'./result/VAE/real_images.npy')


