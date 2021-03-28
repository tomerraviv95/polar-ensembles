import yaml
import torch
import os
from datetime import datetime
import numpy as np


class Entity(object):
    """
    Basic entity, from which trainer and evaluators modules inherit
    implements a few basic methods
    """

    def __init__(self, configuration):
        self.batch_size = 1
        self.config = None
        self.device = None
        self.num_workers = 4
        self.output_dir = ''
        self.iteration = 0
        self.log_dir = None
        self.run_name = ''
        self.val_SNR_start, self.val_SNR_end, self.val_num_SNR = None, None, None
        self.param_parser(configuration)
        self.load_model()
        self.val_SNRs = np.linspace(self.val_SNR_start, self.val_SNR_end, num=self.val_num_SNR)
        self.rand_gen = np.random.RandomState(self.config['noise_seed'])
        self.word_rand_gen = np.random.RandomState(self.config['word_seed'])

    # parsing the config.yaml file
    def param_parser(self, configuration):
        self.config = configuration

        for k, v in self.config.items():
            setattr(self, k, v)

        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(self.output_dir) and len(self.output_dir):
            os.makedirs(self.output_dir)
        if self.log_dir is None:
            self.log_dir = os.path.join(self.output_dir, 'runs')
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.run_name = os.path.join(self.log_dir, self.config['run_name']) if 'run_name' in self.config \
            else os.path.join(self.log_dir, current_time)

        if not os.path.exists(self.run_name):
            os.makedirs(self.run_name)

        if ~os.path.exists(os.path.join(self.run_name, "config.yaml")):
            with open(os.path.join(self.run_name, "config.yaml"), 'w') as file:
                yaml.dump(configuration, file)

    # loading the data
    def data_loading(self):
        pass

    # empty method for loading the model
    def load_model(self):
        pass

    def evaluate(self):
        pass
