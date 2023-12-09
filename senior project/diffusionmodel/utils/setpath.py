import os
import datetime

class Datapath:
    def __init__(self):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.DOCSPATH = './docs'
        self.resultpath = f'./result'
        self.timepath = f'{self.resultpath}/{nowtime}-experiment'
        self.result_trainpath = f'{self.timepath}/training'
        self.RESULT_IMGPATH = f'{self.result_trainpath}/img'

        self.checkpoints_path = f'{self.timepath}/checkpoints'
        self.DIFFUSION_PATH = f"{self.checkpoints_path}/diffusion_model"

        self.logpath = f'{self.timepath}/log'
        self.LOG_CONFIGPATH = f'{self.logpath}/config'
        self.LOG_PREDICTPATH = f'{self.logpath}/pred'
        self.LOG_LOSSPATH = f'{self.logpath}/loss'

        self.path_process()
    def check_folder_path(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def path_process(self):
        self.check_folder_path(self.DOCSPATH)
        self.check_folder_path(self.resultpath)
        self.check_folder_path(self.timepath)
        self.check_folder_path(self.result_trainpath)
        self.check_folder_path(self.RESULT_IMGPATH)
        self.check_folder_path(self.checkpoints_path)
        self.check_folder_path(self.DIFFUSION_PATH)
        self.check_folder_path(self.logpath)
        self.check_folder_path(self.LOG_CONFIGPATH)
        self.check_folder_path(self.LOG_PREDICTPATH)
        self.check_folder_path(self.LOG_LOSSPATH)


