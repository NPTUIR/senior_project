# from models.pix2pix_model import Pix2pix
# from utils import config
# import time
#
# config = config.TrainConfig(epochs=1, #80000
#                             batchsize=1,
#                             gen_lr=0.0002,
#                             dis_lr=0.0002,
#                             g_first_unit=128,
#                             d_first_unit=64)
#
#
#
# gan = Pix2pix(model_name='pix2pix', config=config)
# s = time.time()
# gan.train() # 10000 epochs->used 25826.565 seconds7.17hour
# e = time.time()
# print(f'used {e-s:.4f} seconds')
# gan.predict(training=False)

#--------------------
# from models.CVAE import CVAE
#
# config = config.CVAE_TrainConfig(epochs=400
#                             batchsize=1,
#                             lr=0.0002,
#                             e_first_unit=64,
#                             d_first_unit=512)
# vae = CVAE(model_name='CVAE', config=config)
# vae.train()
#--------------------
from models.cycleGAN_model import Cycle_GAN
cyclegan = Cycle_GAN()
cyclegan.train(epochs=80000, batch_size=1)
# cyclegan.genModel()
