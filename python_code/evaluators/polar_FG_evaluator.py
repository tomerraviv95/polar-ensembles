from python_code.basic.entity import Entity
from python_code.decoders.polar_FG_decoder import PolarFGDecoder
from python_code.dataloaders.channel_dataset import *
from python_code.utils.evaluation_criterion import calculate_accuracy
import time
import torch
import multiprocessing


class PolarFGEvaluator(Entity):
    """
    Wraps the decoder with the evaluation method - pretty straightforward
    """

    def __init__(self, configuration=None):
        self.code_len = None
        self.info_len = None
        self.clipping_val = None
        self.nn_model = "FC"
        self.iteration_num = 5
        self.filter_in_iterations_train = False
        self.filter_in_iterations_eval = False
        self.batches = None
        self.test_errors = None
        self.odd_llr_mask_only = False
        self.even_mask_only = False
        self.multiloss_output_mask_only = False
        self.output_mask_only = False
        self.design_SNR = None
        self.crc = None
        self.code_type = 'Polar'
        self.test_on_zero_word = False
        self.systematic = False

        super().__init__(configuration)
        self.channel_dataset = ChannelModelDataset(code_len=self.code_len, info_len=self.info_len,
                                                   code_type=self.code_type,
                                                   use_llr=True, modulation=BPSKmodulation, channel=AWGN,
                                                   batch_size=self.batch_size, snr_range=self.val_SNRs,
                                                   zero_word_only=self.test_on_zero_word,
                                                   random=self.rand_gen, wordRandom=self.word_rand_gen,
                                                   clipping_val=self.clipping_val, info_ind=self.model.info_ind,
                                                   crc_ind=self.model.crc_ind,
                                                   crc_gm=self.model.crc_gm,
                                                   system_enc=False,
                                                   crc_len=len(self.crc),
                                                   factor_graph=self.model.factor_graph)

        self.dataloaders = torch.utils.data.DataLoader(self.channel_dataset,
                                                       num_workers=min([multiprocessing.cpu_count(), self.num_workers]))

        pass

    def load_model(self, configuration=None):
        self.model = PolarFGDecoder(code_len=self.code_len, info_len=self.info_len, design_SNR=self.design_SNR,
                                    crc=self.crc, iteration_num=self.iteration_num, clipping_val=self.clipping_val,
                                    filter_in_iterations_eval=self.filter_in_iterations_eval,
                                    device=self.device)

    def evaluate(self):
        """
        Evaluation is done at every SNR until a specific number of decoding errors occur
        This ensures more stability at each point, than another method which simply simulates X points at every SNR
        :return: BER and FER vectors
        """
        snr_range = self.val_SNRs
        ber_total, fer_total = np.zeros(len(snr_range), ), np.zeros(len(snr_range), )

        with torch.no_grad():
            for j, snr in enumerate(snr_range):
                err_count = 0
                snr_test_size = 0.0
                batch_empty_flag = False
                print('start eval snr ' + str(snr))
                start = time.time()
                while err_count < self.test_errors and snr_test_size < 10 ** 7:
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
        rx_per_snr = rx_per_snr.to(device=self.device)
        target_per_snr = target_per_snr.to(device=self.device)

        # decode and calculate accuracy
        output_list, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = torch.round(torch.sigmoid(-output_list[-1]))

        ber, fer, err_indices = calculate_accuracy(decoded_words, target_per_snr, self.device)

        ber_total[j] += ber
        fer_total[j] += fer
        err_count += err_indices.shape[0]
        snr_test_size += 1.0

        return err_count, snr_test_size, target_per_snr, rx_per_snr, err_indices, batch_empty_flag
