import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self):
        """
        路徑記得改!!!
        """
        self.realpath = 'C:/Dataset/FacadesDataset/trainA'
        self.sketchpath = 'C:/Dataset/FacadesDataset/trainB'
        self.testrealpath = 'C:/Dataset/FacadesDataset/testA'
        self.testsketchpath = 'C:/Dataset/FacadesDataset/testB'
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
            data_clip = cv2.resize(data, (286, 286), interpolation=cv2.INTER_AREA)
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

            data = np.concatenate([data, data_mirror], axis=0)
            data = np.concatenate([data, data_clip1], axis=0)
            data = np.concatenate([data, data_clip2], axis=0)
            data = np.concatenate([data, data_clip1_mirror], axis=0)
            data = np.concatenate([data, data_clip2_mirror], axis=0)
            return data

    def load_data(self,normalization='[0,1]', data_argumantation=False):
        realdatapath = glob.glob(f'{self.realpath}/*.jpg')#load 400 data
        sketchdatapath = glob.glob(f'{self.sketchpath}/*.jpg')
        self.sketchdata = self.__loadimg(sketchdatapath[0],0,data_argumantation=data_argumantation) #sketchimg.reshape(1, 256, 256, 3)
        self.realdata = self.__loadimg(realdatapath[0],0,data_argumantation=data_argumantation) #realimg.reshape(1, 256, 256, 3)
        for index in range(1,len(realdatapath)):
            self.sketchdata = np.concatenate([self.sketchdata,self.__loadimg(sketchdatapath[index],index,data_argumantation=data_argumantation)],axis=0)
            self.realdata = np.concatenate([self.realdata, self.__loadimg(realdatapath[index],index,data_argumantation=data_argumantation)], axis=0)
            p = int(30*index/400)
            print(f"\rLoading training data: {'*'*p}{'.'*(29-p)} {100*round(index/399,3):.1f}%  ", end='')

        self.realdata = self.realdata.astype(np.float32)
        if normalization == '[0,1]':
            self.realdata = self.realdata/255
        elif normalization == '[-1,1]':
            self.realdata = self.realdata/127.5 - 1
        else:
            raise ValueError('The normalization muse be "[0,1]" or "[-1,1]"')
        self.sketchdata = self.sketchdata.astype(np.float32)
        self.sketchdata = self.sketchdata/255

        print('Loading success!\n')
        return self.sketchdata, self.realdata  #shape=(400, 256, 256, 3)

    def load_testing_data(self):
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
        self.realdata = self.realdata / 255
        self.sketchdata = self.sketchdata.astype(np.float64)
        self.sketchdata = self.sketchdata / 255
        print('Loading success!\n')
        return self.sketchdata, self.realdata  # shape=(400, 256, 256, 3)



if __name__=='__main__':
    # video(speed=2)
    d=DataLoader()
    sketchdata, realdata=d.load_data(normalization='[0,1]',data_argumantation=False);print(sketchdata.shape)
    test_sketchdata, test_realdata = d.load_testing_data();print(test_sketchdata.shape)