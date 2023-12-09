import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def standardization(func):
    def wrapper(*args,**kwargs):
        if kwargs['metrics'] == '[0,1]':
            x, y = func(*args, **kwargs)
            x /= 255
            y /= 255
            return x, y
        elif kwargs['metrics'] == '[-1,1]':
            x, y = func(*args, **kwargs)
            x = x/127.5 - 1
            y = y/127.5 - 1
            return x, y
        else:
            raise ValueError("The metrics parameter only provides '[0,1]' and '[-1,1]' to be entered")
    return wrapper

class DataLoader:
    def __init__(self):
        """
        路徑記得改!!!
        """
        self.realpath = 'F:/Dataset/FacadesDataset/trainA'
        self.sketchpath = 'F:/Dataset/FacadesDataset/trainB'
        self.testrealpath = 'F:/Dataset/FacadesDataset/testA'
        self.testsketchpath = 'F:/Dataset/FacadesDataset/testB'
        self.randomh = np.random.randint(0, 29, size=(400, 2))
        self.randomw = np.random.randint(0, 29, size=(400, 2))
    def __loadimg(self,datapath,index,data_argumantation=True):
        data = cv2.imread(datapath,cv2.IMREAD_COLOR)
        if not data_argumantation:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            return data
        elif data_argumantation:
            h = self.randomh[index]
            w = self.randomw[index]
            data_mirror = cv2.flip(data, flipCode=1) #水平翻轉
            data_clip = cv2.resize(data, (286, 286), interpolation=cv2.INTER_NEAREST)
            data_clip1 = data_clip[h[0]:h[0] + 256, w[0]:w[0] + 256, :]#;print(data0.shape, h, w) # 隨機平移
            data_clip1_mirror = cv2.flip(data_clip1, flipCode=1)  # 隨機平移+水平翻轉
            data_clip2 = data_clip[h[1]:h[1] + 256, w[1]:w[1] + 256, :]  # ;print(data0.shape, h, w) # 隨機平移
            data_clip2_mirror = cv2.flip(data_clip2, flipCode=1)  # 隨機平移+水平翻轉

            data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            data_mirror = cv2.cvtColor(data_mirror, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            data_clip1 = cv2.cvtColor(data_clip1, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            data_clip2 = cv2.cvtColor(data_clip2, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            data_clip1_mirror = cv2.cvtColor(data_clip1_mirror, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            data_clip2_mirror = cv2.cvtColor(data_clip2_mirror, cv2.COLOR_BGR2RGB).reshape(1, 256, 256, 3)
            # plt.imshow(data[0])
            # plt.show()
            # plt.imshow(data_clip1[0])
            # plt.show()
            # plt.imshow(data_clip2[0])
            # plt.show()

            data = np.concatenate([data, data_mirror], axis=0)
            data = np.concatenate([data, data_clip1], axis=0)
            data = np.concatenate([data, data_clip2], axis=0)
            data = np.concatenate([data, data_clip1_mirror], axis=0)
            data = np.concatenate([data, data_clip2_mirror], axis=0)
            #print(data.shape)#(6, 256, 256, 3)

            return data

    def data_argumantation(self,input_data):
        # input_data = cv2.resize(input_data, (286, 286), interpolation=cv2.INTER_NEAREST)
        for i in range(input_data.shape[0]):
            h = np.random.randint(0, 29)
            w = np.random.randint(0, 29)
            input_data[i] = cv2.resize(input_data[i], (286, 286), interpolation=cv2.INTER_NEAREST)
            input_data[i] = input_data[i, h:h+256, w:w+256, :];print(input_data.shape,h,w)

    @standardization
    def load_data(self,metrics='[0,1]',data_argumantation=True):
        realdatapath = glob.glob(f'{self.realpath}/*.jpg')#load 400 data
        sketchdatapath = glob.glob(f'{self.sketchpath}/*.jpg')
        self.sketchdata = self.__loadimg(sketchdatapath[0],0,data_argumantation)#sketchimg.reshape(1, 256, 256, 3)
        self.realdata = self.__loadimg(realdatapath[0],0,data_argumantation)#realimg.reshape(1, 256, 256, 3)
        for index in range(1,len(realdatapath)):
            self.sketchdata = np.concatenate([self.sketchdata,self.__loadimg(sketchdatapath[index],index,data_argumantation)],axis=0)
            self.realdata = np.concatenate([self.realdata, self.__loadimg(realdatapath[index],index,data_argumantation)], axis=0)
            p = int(30*index/400)
            print(f"\rLoading training data: {'*'*p}{'.'*(29-p)} {100*round(index/399,3):.1f}%  ", end='')

        self.realdata = self.realdata.astype(np.float32)
        self.sketchdata = self.sketchdata.astype(np.float32)

        print('Loading success!\n')
        return self.sketchdata, self.realdata  #shape=(400, 256, 256, 3)

    @standardization
    def load_testing_data(self,metrics='[0,1]'):
        realdatapath = glob.glob(f'{self.testrealpath}/*.jpg')  # load 400 data
        sketchdatapath = glob.glob(f'{self.testsketchpath}/*.jpg')
        self.sketchdata = self.__loadimg(sketchdatapath[0],0,data_argumantation=False)  # sketchimg.reshape(1, 256, 256, 3)
        self.realdata = self.__loadimg(realdatapath[0],0,data_argumantation=False)  # realimg.reshape(1, 256, 256, 3)
        for index in range(1, len(realdatapath)):
            self.sketchdata = np.concatenate([self.sketchdata, self.__loadimg(sketchdatapath[index],index,data_argumantation=False)], axis=0)
            self.realdata = np.concatenate([self.realdata, self.__loadimg(realdatapath[index],index,data_argumantation=False)], axis=0)
            p = int(30 * index / 106)
            print(f"\rLoading testing data: {'*' * p}{'.' * (29 - p)} {100 * round(index / 105, 3):.1f}%  ", end='')

        self.realdata = self.realdata.astype(np.float64)
        self.sketchdata = self.sketchdata.astype(np.float64)
        print('Loading success!\n')
        return self.sketchdata, self.realdata  # shape=(400, 256, 256, 3)



if __name__=='__main__':
    # video(speed=2)
    d=DataLoader()
    sketchdata, realdata = d.load_data(metrics='[0,1]');print(sketchdata.shape)
    test_sketchdata, test_realdata = d.load_testing_data(metrics='[0,1]');print(test_sketchdata.shape)

