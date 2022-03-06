from python_code.utils.python_utils import llr_to_bits
from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.data.channel_model import BPSKmodulation, AWGN
import torch
from torch.utils import data
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.decoders.ensemble_decoder import EnsembleDecoder
from python_code.trainers.trainer import Trainer
from python_code.data.dataset_crc import DatasetCRC
from python_code.data.channel_dataset import ChannelModelDataset
from globals import CONFIG, DEVICE
import time
import numpy as np

EARLY_TERMINATION = True
SYSTEMATIC_ENCODING = False
USE_LLR = True


class EnsembleTrainer(Trainer):
    """
    Wraps the decoder with the evaluation method
    """
    def __init__(self,test_mode=False):
        self.test_mode = test_mode
        super().__init__()

    def load_model(self):
        self.model = EnsembleDecoder(code_len=CONFIG.code_len,
                                     info_len=CONFIG.info_len,
                                     design_snr=CONFIG.design_SNR,
                                     iteration_num=CONFIG.iteration_num,
                                     clipping_val=CONFIG.clipping_val,
                                     device=DEVICE,
                                     crc_order=CONFIG.crc_order,
                                     num_of_decoders=CONFIG.ensemble_dec_num,
                                     ensemble_crc_dist=CONFIG.ensemble_crc_dist)
        self.decoder_name = 'Ensemble'

    def setup_dataloader(self):
        rand_gen = np.random.RandomState(CONFIG.noise_seed)
        word_rand_gen = np.random.RandomState(CONFIG.word_seed)
        train_SNRs = np.linspace(CONFIG.train_SNR_start, CONFIG.train_SNR_end, num=CONFIG.train_num_SNR)
        val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
        zero_word_only = {'train': True, 'val': False}
        self.snr_range = {'train': train_SNRs, 'val': val_SNRs}
        batch_size = {'train': CONFIG.train_minibatch_size, 'val': CONFIG.val_batch_size}
        self.channel_dataset = {}

        self.channel_dataset['train'] = DatasetCRC(load_dataset=CONFIG.load_dataset,
                                                  save_dataset=False,
                                                  words_per_crc=CONFIG.words_per_crc_range,
                                                  code_len=CONFIG.code_len,
                                                  info_len=CONFIG.info_len,
                                                  code_type=CONFIG.code_type,
                                                  use_llr=USE_LLR,
                                                  modulation=BPSKmodulation,
                                                  channel=AWGN,
                                                  batch_size=batch_size['train'],
                                                  snr_range=self.snr_range['train'],
                                                  zero_word_only=zero_word_only['train'],
                                                  random=rand_gen,
                                                  wordRandom=word_rand_gen,
                                                  clipping_val=CONFIG.clipping_val,
                                                  info_ind=self.model.info_ind,
                                                  system_enc=SYSTEMATIC_ENCODING,
                                                  code_gm=self.model.code_gm,
                                                  decoder_name=self.decoder_name,
                                                  crc_order=CONFIG.crc_order)

        if self.test_mode:
            self.channel_dataset['val'] = ChannelModelDataset(code_len=CONFIG.code_len,
                                                               info_len=CONFIG.info_len,
                                                               code_type=CONFIG.code_type,
                                                               use_llr=USE_LLR,
                                                               modulation=BPSKmodulation,
                                                               channel=AWGN,
                                                               batch_size=batch_size['val'],
                                                               snr_range=self.snr_range['val'],
                                                               zero_word_only=zero_word_only['val'],
                                                               random=rand_gen,
                                                               wordRandom=word_rand_gen,
                                                               clipping_val=CONFIG.clipping_val,
                                                               info_ind=self.model.info_ind,
                                                               system_enc=SYSTEMATIC_ENCODING,
                                                               code_gm=self.model.code_gm,
                                                               decoder_name=self.decoder_name,
                                                               crc_order=CONFIG.crc_order)
        else:
            self.channel_dataset['val'] = DatasetCRC(load_dataset=CONFIG.load_dataset,
                                         save_dataset=False,
                                         words_per_crc=CONFIG.words_per_crc_range,
                                         code_len=CONFIG.code_len,
                                         info_len=CONFIG.info_len,
                                         code_type=CONFIG.code_type,
                                         use_llr=USE_LLR,
                                         modulation=BPSKmodulation,
                                         channel=AWGN,
                                         batch_size=batch_size['val'],
                                         snr_range=self.snr_range['val'],
                                         zero_word_only=zero_word_only['val'],
                                         random=rand_gen,
                                         wordRandom=word_rand_gen,
                                         clipping_val=CONFIG.clipping_val,
                                         info_ind=self.model.info_ind,
                                         system_enc=SYSTEMATIC_ENCODING,
                                         code_gm=self.model.code_gm,
                                         decoder_name=self.decoder_name,
                                         crc_order=CONFIG.crc_order)

        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase]) for phase in
                            ['train', 'val']}


    def calc_loss(self, prediction, labels):
        output, not_satisfied_list = prediction
        return self.criterion(-output, labels)

    def decode(self, soft_values):
        return llr_to_bits(soft_values)

    def test(self):
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
                    ber, fer, err_indices = self.single_test(j)
                    ber_total[j] += ber
                    fer_total[j] += fer
                    err_count += err_indices.shape[0]
                    snr_test_size += 1.0

                ber_total[j] /= snr_test_size
                fer_total[j] /= snr_test_size
                print(
                    f'done. time: {time.time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}, log-ber:{-np.log(ber_total[j])}')
            return ber_total, fer_total

    def single_test(self, j):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output_list, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = self.decode(output_list[-1])

        return calculate_accuracy(decoded_words, target_per_snr, DEVICE)

    def evaluate(self):
        """
        Evaluation is done at every SNR until a specific number of decoding errors occur
        This ensures more stability at each point, than another method which simply simulates X points at every SNR
        :return: BER and FER vectors
        """

        if self.test_mode:
            ber_total, fer_total = self.test()
            return ber_total, fer_total

        snr_range = self.snr_range['val']
        ber_total, fer_total = np.zeros(len(snr_range)), np.zeros(len(snr_range))
        with torch.no_grad():
            # for j, snr in enumerate(snr_range):
            # print('start eval snr ' + str(snr))
            start = time.time()
            cont_dataset = self.channel_dataset['val'].getContinues()
            max_batch = round(len(cont_dataset)/CONFIG.val_batch_size)
            val_loader = data.DataLoader(cont_dataset, batch_size=CONFIG.val_batch_size, shuffle=False)
            i=0
            for rx, target, crc_val in val_loader:
                ber, fer, err_indices = self.single_eval(rx, target, crc_val)
                ber_total += ber
                fer_total += fer
                i+=1
                print(f"val batch {i}/{max_batch}")

            print(f'done. time: {time.time() - start}, ber: {ber_total}, fer: {fer_total}, log-ber:{-np.log(ber_total)}')
        return ber_total, fer_total

    def single_eval(self, rx_per_snr, target_per_snr, crc_val):
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = self.decode(output)

        return calculate_accuracy(decoded_words, target_per_snr, DEVICE)

    def get_rx_target_crc(self,data):
        recieved = data[:, :CONFIG.code_len]
        target = data[:, CONFIG.code_len:(CONFIG.code_len+CONFIG.info_len)]
        crc_val = data[:, -1]

        return recieved, target, crc_val

    def train(self):
        self.optimization_setup()
        self.loss_setup()
        snr_range = self.snr_range['train']
        self.evaluate()
        ber_total, fer_total, best_ber = 1, 1, 1
        early_stopping_bers = []
        dataset_size = self.channel_dataset['train'].dataset_size
        idx = np.array(range(dataset_size[0]))
        batch_size = CONFIG.train_minibatch_size
        losses = [0]*(CONFIG.num_of_epochs+1)
        for epoch in range(1, CONFIG.num_of_epochs + 1):
            print(f'Epoch {epoch}')
            loss_iter_num = 0
            train_cont_dataset = self.channel_dataset['train'].getContinues()
            max_batch = round(len(train_cont_dataset) / CONFIG.train_minibatch_size)
            train_loader = data.DataLoader(train_cont_dataset, batch_size=CONFIG.train_minibatch_size, shuffle=True)
            for rx, target, crc_val in train_loader:
                prediction = self.model(rx)
                # calculate loss
                loss = self.calc_loss(prediction=prediction, labels=target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses[epoch] += loss
                loss_iter_num += 1
                print(f"train batch {loss_iter_num}/{max_batch}")

            losses[epoch] /= loss_iter_num
                # rx_per_snr, target_per_snr, crc_val_per_snr = iter(self.channel_dataset['train'][j])
                # rx_per_snr = rx_per_snr.to(device=DEVICE)
                # target_per_snr = target_per_snr.to(device=DEVICE)
                # # shuffle the data
                # rx_per_snr = rx_per_snr[idx]
                # target_per_snr = target_per_snr[idx]
                #
                # for i in range(int(dataset_size[0]/batch_size)):
                #     if ((i+1)*batch_size >= dataset_size[0]):
                #         break
                #     rx = rx_per_snr[i*batch_size:(i+1)*batch_size]
                #     target = target_per_snr[i*batch_size:(i+1)*batch_size]
                #     prediction = self.model(rx)
                #     # calculate loss
                #     loss = self.calc_loss(prediction=prediction, labels=target)
                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     self.optimizer.step()

            if epoch % (CONFIG.validation_epochs) == 0:
                prev_ber_total = ber_total
                ber_total, fer_total = self.evaluate()
                print(f"train_loss epoch {epoch} : {losses[epoch]}")
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

if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = EnsembleTrainer()

    start = time.time()

    ber, fer = dec.train()

    end = time.time()
    print(f'################## total training time: {end-start} ##################')
