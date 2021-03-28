from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam, RMSprop

from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.data.channel_model import BPSKmodulation, AWGN
from python_code.data.channel_dataset import ChannelModelDataset
from globals import CONFIG, DEVICE
import numpy as np
import torch
import time


class Trainer(object):
    """
    Basic entity, from which trainer and trainers modules inherit
    implements a few basic methods
    """

    def __init__(self):
        self.load_model()
        self.setup_dataloader()

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
                                                           use_llr=True,
                                                           modulation=BPSKmodulation,
                                                           channel=AWGN,
                                                           batch_size=batch_size[phase],
                                                           snr_range=self.snr_range[phase],
                                                           zero_word_only=zero_word_only[phase],
                                                           random=rand_gen,
                                                           wordRandom=word_rand_gen,
                                                           clipping_val=CONFIG.clipping_val,
                                                           info_ind=self.model.info_ind,
                                                           crc_ind=self.model.crc_ind,
                                                           crc_gm=self.model.crc_gm,
                                                           system_enc=CONFIG.systematic_encoding,
                                                           crc_len=len(CONFIG.crc),
                                                           factor_graph=self.model.factor_graph)
                                for phase in ['train', 'val']}
        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase]) for phase in
                            ['train', 'val']}

    # empty method for loading the model
    def load_model(self):
        self.model = None

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
                print(f'done. time: {time.time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}')
            return ber_total, fer_total

    def single_eval(self, j):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output_list, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = torch.round(torch.sigmoid(-output_list[-1]))

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
    def calc_loss(self, decision, labels):
        return self.criterion(decision, labels)

    ############
    # Training #
    ############
    def train(self):
        self.optimization_setup()
        self.loss_setup()
        snr_range = self.snr_range['train']
        self.evaluate()
        for epoch in range(1, CONFIG.num_of_epochs + 1):
            print(f'Epoch {epoch}')
            for j, snr in enumerate(snr_range):
                # draw train data
                rx_per_snr, target_per_snr = iter(self.channel_dataset['train'][j])
                rx_per_snr = rx_per_snr.to(device=DEVICE)
                target_per_snr = target_per_snr.to(device=DEVICE)

                output_list, not_satisfied_list = self.model(rx_per_snr)

                # calculate loss
                loss = self.calc_loss(decision=output_list[-1], labels=target_per_snr)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % (CONFIG.validation_epochs) == 0:
                self.evaluate()