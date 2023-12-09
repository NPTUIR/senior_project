import datetime
import os

class Path:
    def __init__(self,model_name=''):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.CONFIGPATH = f'./result/{model_name}/{nowtime}/config'
        self.PROCESSPATH = f'./result/{model_name}/{nowtime}/training_process'
        self.LOSSPATH = f'./result/{model_name}/{nowtime}/loss'
        self.MODELPATH = f'./result/{model_name}/{nowtime}/model'
        self.H5PATH = f'./result/training/h5'
        self.TRAININGIMGPATH = f'./result/training/img'
        self.__make_buffer_folder()
    def __make_buffer_folder(self):
        if not os.path.exists(self.H5PATH):
            os.makedirs(self.H5PATH)
        if not os.path.exists(self.TRAININGIMGPATH):
            os.makedirs(self.TRAININGIMGPATH)
    def make_folder(self):
        for path in [self.CONFIGPATH, self.PROCESSPATH, self.LOSSPATH, self.MODELPATH]:
            if not os.path.exists(path):
                os.makedirs(path)