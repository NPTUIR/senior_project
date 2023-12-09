import json

class TrainingConfig:
    def __init__(self,
                 dataset_name="oxford_flowers102",
                 dataset_repetitions = 5,
                 num_epochs = 50,  # train for at least 50 epochs for good results
                 image_size = 64,
                 kid_image_size = 75,
                 kid_diffusion_steps = 5,
                 plot_diffusion_steps = 20,
                 min_signal_rate = 0.02,
                 max_signal_rate = 0.95,
                 embedding_dims = 32,
                 embedding_max_frequency = 1000.0,
                 batch_size = 64,
                 ema = 0.999,
                 learning_rate = 1e-3,
                 weight_decay = 1e-4,
                ):
        self.dataset_name = dataset_name
        self.dataset_repetitions = dataset_repetitions
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.kid_image_size = kid_image_size
        self.kid_diffusion_steps = kid_diffusion_steps
        self.plot_diffusion_steps = plot_diffusion_steps
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.embedding_dims = embedding_dims
        self.embedding_max_frequency = embedding_max_frequency
        self.batch_size = batch_size
        self.ema = ema
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def save_to_json(self, path='./config.json'):
        keys = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        values = [self.batch_size, self.dataset_name, self.dataset_repetitions, self.ema, self.embedding_dims, self.embedding_max_frequency, self.image_size, self.kid_diffusion_steps, self.kid_image_size, self.learning_rate, self.max_signal_rate, self.min_signal_rate, self.num_epochs, self.plot_diffusion_steps, self.weight_decay]
        # print(keys)
        data = dict(zip(keys, values))
        if path == './config.json':
            print(f'\033[4;31m[WARNNING] json file is not in real path.\033[0m')
        with open(path, 'w') as f:
            json.dump(data, f)

if __name__=='__main__':
    t = TrainingConfig()
    t.save_to_json()