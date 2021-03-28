from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.data.channel_model import BPSKmodulation, AWGN
from python_code.data.channel_dataset import ChannelModelDataset
from globals import CONFIG, DEVICE
import numpy as np
import torch
import time

MAX_TEST_SIZE = 10 ** 7


class Trainer(object):
    """
    Basic entity, from which trainer and trainers modules inherit
    implements a few basic methods
    """

    def __init__(self):
        self.load_model()
        rand_gen = np.random.RandomState(CONFIG.noise_seed)
        word_rand_gen = np.random.RandomState(CONFIG.word_seed)
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        self.channel_dataset = ChannelModelDataset(code_len=CONFIG.code_len,
                                                   info_len=CONFIG.info_len,
                                                   code_type=CONFIG.code_type,
                                                   use_llr=True,
                                                   modulation=BPSKmodulation,
                                                   channel=AWGN,
                                                   batch_size=CONFIG.batch_size,
                                                   snr_range=val_SNRs,
                                                   zero_word_only=CONFIG.test_on_zero_word,
                                                   random=rand_gen,
                                                   wordRandom=word_rand_gen,
                                                   clipping_val=CONFIG.clipping_val,
                                                   info_ind=self.model.info_ind,
                                                   crc_ind=self.model.crc_ind,
                                                   crc_gm=self.model.crc_gm,
                                                   system_enc=False,
                                                   crc_len=len(CONFIG.crc),
                                                   factor_graph=self.model.factor_graph)
        self.dataloaders = torch.utils.data.DataLoader(self.channel_dataset)

    # empty method for loading the model
    def load_model(self):
        self.model = None

    def evaluate(self):
        """
        Evaluation is done at every SNR until a specific number of decoding errors occur
        This ensures more stability at each point, than another method which simply simulates X points at every SNR
        :return: BER and FER vectors
        """
        snr_range = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        ber_total, fer_total = np.zeros(len(snr_range), ), np.zeros(len(snr_range), )

        with torch.no_grad():
            for j, snr in enumerate(snr_range):
                err_count = 0
                snr_test_size = 0.0
                batch_empty_flag = False
                print('start eval snr ' + str(snr))
                start = time.time()
                while err_count < CONFIG.test_errors and snr_test_size < MAX_TEST_SIZE:
                    eval_output = self.single_eval(ber_total, err_count, fer_total, j, snr_test_size, batch_empty_flag)
                    err_count, snr_test_size, target_per_snr, rx_per_snr, err_indices, batch_empty_flag = eval_output
                    if batch_empty_flag:
                        break

                ber_total[j] /= snr_test_size
                fer_total[j] /= snr_test_size
                print(f'done. time: {time.time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}')
            return ber_total, fer_total

    def single_eval(self, ber_total, err_count, fer_total, j, snr_test_size, batch_empty_flag):
        # create test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset[j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output_list, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = torch.round(torch.sigmoid(-output_list[-1]))

        ber, fer, err_indices = calculate_accuracy(decoded_words, target_per_snr, DEVICE)

        ber_total[j] += ber
        fer_total[j] += fer
        err_count += err_indices.shape[0]
        snr_test_size += 1.0

        return err_count, snr_test_size, target_per_snr, rx_per_snr, err_indices, batch_empty_flag
