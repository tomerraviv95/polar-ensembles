from dir_definitions import WEIGHTS_DIR, CONFIG_PATH
from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.data.channel_model import BPSKmodulation, AWGN
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, RMSprop
from python_code.data.channel_dataset import ChannelModelDataset
from globals import CONFIG, DEVICE
from shutil import copyfile
import numpy as np
import torch
import time
import os

EARLY_STOPPING_PATIENCE = 5
SYSTEMATIC_ENCODING = False
USE_LLR = True


class Trainer(object):
    """
    Basic entity, from which trainer and trainers modules inherit
    implements a few basic methods
    """

    def __init__(self):
        self.load_model()
        self.setup_dataloader()
        self.setup_save_dir()

        if CONFIG.load_weights:
            self.load_weights()

    def setup_dataloader(self):
        rand_gen = np.random.RandomState(CONFIG.noise_seed)
        word_rand_gen = np.random.RandomState(CONFIG.word_seed)
        train_SNRs = np.linspace(CONFIG.train_SNR_start, CONFIG.train_SNR_end, num=CONFIG.train_num_SNR)
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        zero_word_only = {'train': True, 'val': False}
        self.snr_range = {'train': train_SNRs, 'val': val_SNRs}
        batch_size = {'train': CONFIG.train_minibatch_size, 'val': CONFIG.val_batch_size}
        self.channel_dataset = {phase: ChannelModelDataset(code_len=CONFIG.code_len,
                                                           info_len=CONFIG.info_len,
                                                           code_type=CONFIG.code_type,
                                                           use_llr=USE_LLR,
                                                           modulation=BPSKmodulation,
                                                           channel=AWGN,
                                                           batch_size=batch_size[phase],
                                                           snr_range=self.snr_range[phase],
                                                           zero_word_only=zero_word_only[phase],
                                                           random=rand_gen,
                                                           wordRandom=word_rand_gen,
                                                           clipping_val=CONFIG.clipping_val,
                                                           info_ind=self.model.info_ind,
                                                           system_enc=SYSTEMATIC_ENCODING,
                                                           code_gm=self.model.code_gm,
                                                           decoder_name=self.decoder_name)
                                for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase]) for phase in
                            ['train', 'val']}

    # empty method for loading the model
    def load_model(self):
        self.model = None
        self.decoder_name = None

    def decode(self, soft_values):
        pass

    def setup_save_dir(self):
        self.weights_dir = os.path.join(WEIGHTS_DIR, CONFIG.run_name)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            # save config in output dir
            copyfile(CONFIG_PATH, os.path.join(self.weights_dir, "config.yaml"))

    def evaluate(self):
        """
        Evaluation is done at every SNR until a specific number of decoding errors occur
        This ensures more stability at each point, than another method which simply simulates X points at every SNR
        :return: BER and FER vectors
        """
        snr_range = self.snr_range['val']
        ber_total, fer_total = np.zeros(len(snr_range)), np.zeros(len(snr_range))

        with torch.no_grad():
            for j, snr in enumerate(snr_range):
                err_count = 0
                snr_test_size = 0.0
                print('start eval snr ' + str(snr))
                start = time.time()
                while err_count < CONFIG.test_errors:
                    ber, fer, err_indices = self.single_eval(j)
                    ber_total[j] += ber
                    fer_total[j] += fer
                    err_count += err_indices.shape[0]
                    snr_test_size += 1.0

                ber_total[j] /= snr_test_size
                fer_total[j] /= snr_test_size
                print(
                    f'done. time: {time.time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}, log-ber:{-np.log(ber_total[j])}')
            return ber_total, fer_total

    def single_eval(self, j):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output_list, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = self.decode(output_list[-1])

        return calculate_accuracy(decoded_words, target_per_snr, DEVICE)

    def optimization_setup(self):
        if CONFIG.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=CONFIG.lr)
        elif CONFIG.optimizer_type == 'ADAM':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=CONFIG.lr)
        elif CONFIG.optimizer_type == 'RMSPROP':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()), lr=CONFIG.lr)
        else:
            raise ValueError('No such optimizer type!')

    def loss_setup(self):
        if CONFIG.criterion_type == 'BCE':
            self.criterion = BCEWithLogitsLoss().to(device=DEVICE)
        else:
            raise ValueError('No such loss type!')

    # calculate train loss
    def calc_loss(self, prediction, labels):
        return self.criterion(prediction, labels)

    def check_early_stopping(self, ber, prev_ber, early_stopping_bers):
        if ber > prev_ber:
            early_stopping_bers.append(0)

        if len(early_stopping_bers) > EARLY_STOPPING_PATIENCE:
            return True
        return False

    def save_weights(self, epoch):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch},
                   os.path.join(self.weights_dir, f'epoch_{epoch}.pt'))

    def load_weights(self):
        """
        Loads detector's weights defined by the [snr,gamma] from checkpoint, if exists
        """
        if os.path.isdir(self.weights_dir):
            files = os.listdir(self.weights_dir)
            names = []
            for file in files:
                if file.startswith("epoch_"):
                    names.append(int(file.split('.')[0].split('_')[1]))
            names.sort()
            print(f'loading model from epoch {names[-1]}')
            checkpoint = torch.load(os.path.join(self.weights_dir, 'epoch_' + str(names[-1]) + '.pt'))
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except Exception:
                raise ValueError("Wrong run directory!!!")
        else:
            print(f'No such dir!!! starting from scratch')

    ############
    # Training #
    ############
    def train(self):
        self.optimization_setup()
        self.loss_setup()
        snr_range = self.snr_range['train']
        self.evaluate()
        ber_total, fer_total, best_ber = 1, 1, 1
        early_stopping_bers = []
        for epoch in range(1, CONFIG.num_of_epochs + 1):
            print(f'Epoch {epoch}')
            for j, snr in enumerate(snr_range):
                # draw train data
                rx_per_snr, target_per_snr = iter(self.channel_dataset['train'][j])
                rx_per_snr = rx_per_snr.to(device=DEVICE)
                target_per_snr = target_per_snr.to(device=DEVICE)

                prediction = self.model(rx_per_snr)

                # calculate loss
                loss = self.calc_loss(prediction=prediction, labels=target_per_snr)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % (CONFIG.validation_epochs) == 0:
                prev_ber_total = ber_total
                ber_total, fer_total = self.evaluate()

                # extract relevant ber, either scalar or last value in list
                if type(ber_total) == list:
                    raise ValueError('Must run training with single eval SNR!!!')

                ber, prev_ber = ber_total, prev_ber_total

                # save weights if model is improved compared to best ber
                if ber < best_ber:
                    self.save_weights(epoch)
                    best_ber = ber

                # early stopping
                if CONFIG.early_stopping and self.check_early_stopping(ber, prev_ber, early_stopping_bers):
                    break

        return ber_total, fer_total
