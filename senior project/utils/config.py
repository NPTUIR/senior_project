import json

class TrainConfig:
    def __init__(self,
                 epochs:int = 10000,
                 batchsize:int = 32,
                 gen_lr:float = 0.0002,
                 dis_lr:float = 0.0002,
                 lambda_:int = 100,
                 g_first_unit:int = 128,
                 d_first_unit:int = 64,
                 ):
        self.batchsize = batchsize
        self.epochs = epochs
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.lambda_ = lambda_
        self.g_first_unit = g_first_unit
        self.d_first_unit = d_first_unit

    def save_to_json(self, used_time, path='./config.json'):
        # keys = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        keys = ['batchsize','dis_lr','epochs','gen_lr','lambda_','g_first_unit','d_first_unit','used_time']
        values = [self.batchsize,self.dis_lr,self.epochs,self.gen_lr,self.lambda_,self.g_first_unit,self.d_first_unit, used_time]
        # print(keys)
        data = dict(zip(keys, values))
        if path=='./config.json':
            print(f'\033[4;31m[WARNNING] json file is not in real path.\033[0m')
        with open(path,'w') as f:
            json.dump(data,f)

class CVAE_TrainConfig:
    def __init__(self,
                 epochs:int = 10000,
                 batchsize:int = 32,
                 lr:float = 0.0002,
                 e_first_unit:int = 64,
                 d_first_unit:int = 512,
                 ):
        self.batchsize = batchsize
        self.epochs = epochs
        self.lr = lr
        self.e_first_unit = e_first_unit
        self.d_first_unit = d_first_unit

    def save_to_json(self, used_time, path='./config.json'):
        # keys = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        keys = ['batchsize','epochs','lr','e_first_unit','d_first_unit','used_time']
        values = [self.batchsize,self.epochs,self.lr,self.e_first_unit,self.d_first_unit, used_time]
        # print(keys)
        data = dict(zip(keys, values))
        if path=='./config.json':
            print(f'\033[4;31m[WARNNING] json file is not in real path.\033[0m')
        with open(path,'w') as f:
            json.dump(data,f)

if __name__=='__main__':
    t = TrainConfig()
    t.save_to_json()