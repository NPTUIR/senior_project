import glob, os, cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization


def get_transform_data(model_name='none',
                       model_path='.\VAE_model.h5',
                       dir=r'.\sketchdata\*.jpg'):
    # 路徑要改
    dir = glob.glob(dir)
    # h5模型要改一下
    model = load_model(model_path)
    dataname = [f'1{i}_B' for i in range(10)] + ['1_B'] + [f'2{i}_B' for i in range(10)] + ['2_B'] + ['30_B', '31_B',
                                                                                                      '32_B'] + [
                   f'{i}_B' for i in range(3, 10)]
    for i in range(32):
        datapath = dir[i]
        data = cv2.imread(datapath, cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
        data = data.astype(np.float32) / 255

        pred = model.predict(data)
        pred = (pred + 1) / 2
        if os.path.exists(f'web result/{model_name}/'):
            os.makedirs(f'web result/{model_name}/')
        plt.imsave(f'web result/{model_name}/{dataname[i]}.jpg', pred[0])
get_transform_data()