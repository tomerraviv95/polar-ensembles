import os.path

from dir_definitions import DATABASE_DIR
from globals import CONFIG, DEVICE
from python_code.data.channel_model import BPSKmodulation, AWGN
from python_code.codes.crc import crc
from python_code.data.channel_dataset import ChannelModelDataset
from python_code.decoders.ensemble_decoder import EnsembleDecoder
from python_code.decoders.fg_decoder import FGDecoder
import concurrent.futures
import collections
import numpy as np
import torch
from python_code.utils.python_utils import llr_to_bits


SYSTEMATIC_ENCODING = False


class DatasetCRC(ChannelModelDataset):
    '''
    generate dataset with uniform crc_range distribution over each SNR
    database is per snr, for each snr the data is tensor where first code_len elements are the received bits,
    followed by info_len elements which are the original bit transmitted and last element is the BP output crc val
    '''
    def __init__(self, load_dataset, save_dataset, words_per_crc, code_len, info_len, code_type,
                 clipping_val, decoder_name, use_llr=True,
                 modulation=BPSKmodulation,
                 channel=AWGN, batch_size=None,
                 snr_range=None, zero_word_only=True,
                 random=None, wordRandom=None,
                 crc_order=0,
                 **code_params):

        super().__init__(code_len=code_len, info_len=info_len, code_type=code_type,
                  clipping_val=clipping_val, decoder_name=decoder_name, use_llr=use_llr,
                  modulation=modulation,
                  channel=channel, batch_size=batch_size,
                  snr_range=snr_range, zero_word_only=zero_word_only,
                  random=random, wordRandom=wordRandom,
                  crc_order=crc_order,
                  info_ind=code_params["info_ind"],
                  system_enc=code_params["system_enc"],
                  code_gm=code_params["code_gm"]
                  )

        self.load_dataset = load_dataset
        self.save_dataset = save_dataset
        self.words_per_crc = words_per_crc
        self.decoder_name = decoder_name
        self.database = []
        for i in range(self.snr_range.size):
            self.database.append([])
        self.load_model()
        self.crc2int = lambda crc : int("".join(str(int(x)) for x in crc),2)


        if load_dataset:
            self.load_data()
        else:
            self.generate_data()

    def load_model(self):
        if self.decoder_name == "Ensemble":
            self.model = EnsembleDecoder(code_len=self.code_len,
                                         info_len=self.info_len,
                                         design_snr=CONFIG.design_SNR,
                                         iteration_num=CONFIG.iteration_num,
                                         clipping_val=self.clipping_val,
                                         device=DEVICE,
                                         crc_order=self.crc_order,
                                         num_of_decoders=CONFIG.ensemble_dec_num,
                                         ensemble_crc_dist=CONFIG.ensemble_crc_dist)

        elif self.decoder_name == "FG":
            print(f"model {self.decoder_name} is not supported")
            raise ValueError
            # self.model = FGDecoder(code_len=self.code_len,
            #                        info_len=self.info_len,
            #                        design_snr=CONFIG.design_SNR,
            #                        iteration_num=CONFIG.iteration_num,
            #                        clipping_val=self.clipping_val,
            #                        device=DEVICE)

    def generate_data(self):
        if CONFIG.ensemble_crc_dist == "uniform":
            words_per_crc_counter = np.zeros([len(self.snr_range),CONFIG.ensemble_dec_num])
        self.channel_dataset = ChannelModelDataset(code_len=self.code_len,
                                                           info_len=self.info_len,
                                                           code_type='Polar',
                                                           use_llr=self.use_llr,
                                                           modulation=self.modulation,
                                                           channel=self.channel,
                                                           batch_size=self.batch_size,
                                                           snr_range=self.snr_range,
                                                           zero_word_only=self.zero_word_only,
                                                           random=self.random,
                                                           wordRandom=self.wordRandom,
                                                           clipping_val=self.clipping_val,
                                                           info_ind=self.model.info_ind,
                                                           system_enc=SYSTEMATIC_ENCODING,
                                                           code_gm=self.model.code_gm,
                                                           decoder_name=self.decoder_name,
                                                           crc_order=self.crc_order)

        words = 0
        j = 0
        done = False
        while j < self.snr_range.size:
            while not(done):
                # draw test data
                rx_per_snr, target_per_snr = iter(self.channel_dataset[j])
                rx_per_snr = rx_per_snr.to(device=DEVICE)
                target_per_snr = target_per_snr.to(device=DEVICE)

                # decode and calculate accuracy
                output, not_satisfied_list = self.model(rx_per_snr)
                decoded_words = self.decode(output)
                pred_crc = crc.crc_check(decoded_words, self.crc_order)

                for val in pred_crc:
                    if (np.sum(words_per_crc_counter[j]) >= 4*self.words_per_crc): # dataset is full
                        self.database[j] = torch.Tensor(self.database[j])
                        if self.save_dataset:
                            self.save_data(j,self.snr_range[j])
                        j += 1
                        if j >= self.snr_range.size:
                            done = True
                            break
                    words +=1
                    val_int = self.crc2int(val)
                    if val_int == 0:
                        continue
                    idx = int(val_int//((2**self.crc_order)/CONFIG.ensemble_dec_num))
                    if words_per_crc_counter[j,idx] >= self.words_per_crc:
                        continue
                    val_tens = torch.Tensor([val_int])
                    self.database[j].append(torch.cat((rx_per_snr[idx],target_per_snr[idx], val_tens),0))
                    words_per_crc_counter[j,idx] += 1
                    print(f"snr {j+1}/{len(self.snr_range)} words {words}, per range {words_per_crc_counter[j]}")




    def load_data(self):
        for j,snr in enumerate(self.snr_range):
            self.database[j] = torch.load(os.path.join(f'{DATABASE_DIR}\\dataset_code_len_{self.code_len}_crc_order_{self.crc_order}_snr_{snr}.pt'))

    def save_data(self,j,snr):
        torch.save(self.database[j], os.path.join(f'{DATABASE_DIR}\\dataset_code_len_{self.code_len}_crc_order_{self.crc_order}_snr_{snr}.pt'))


    def __getitem__(self, item):
        snr_data = self.database[item]
        recieved = snr_data[:][:self.code_len]
        target = snr_data[:][self.code_len:(self.code_len+self.info_len)]
        crc_val = snr_data[:][-1]
        recieved = torch.tensor(recieved).float().view(-1, self.code_len)
        target = torch.tensor(target).float().view(-1, self.info_len)
        return recieved, target

    def decode(self, soft_values):
        return llr_to_bits(soft_values)


if __name__ == "__main__":

    model = EnsembleDecoder(code_len=CONFIG.code_len,
                            info_len=CONFIG.info_len,
                            design_snr=CONFIG.design_SNR,
                            iteration_num=CONFIG.iteration_num,
                            clipping_val=CONFIG.clipping_val,
                            device=DEVICE,
                            crc_order=CONFIG.crc_order,
                            num_of_decoders=CONFIG.ensemble_dec_num,
                            ensemble_crc_dist=CONFIG.ensemble_crc_dist)

    rand_gen = np.random.RandomState(CONFIG.noise_seed)
    word_rand_gen = np.random.RandomState(CONFIG.word_seed)
    train_SNRs = np.linspace(CONFIG.train_SNR_start, CONFIG.train_SNR_end, num=CONFIG.train_num_SNR)
    val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)
    zero_word_only = {'train': True, 'val': False}
    snr_range = train_SNRs

    data = DatasetCRC(load_dataset=False, save_dataset=True, words_per_crc=CONFIG.words_per_crc_range,
                      code_len=CONFIG.code_len,
                      info_len=CONFIG.info_len,
                      code_type=CONFIG.code_type,
                      use_llr=True,
                      modulation=BPSKmodulation,
                      channel=AWGN,
                      batch_size=CONFIG.train_minibatch_size,
                      snr_range=snr_range,
                      zero_word_only=False,
                      random=rand_gen,
                      wordRandom=word_rand_gen,
                      clipping_val=CONFIG.clipping_val,
                      info_ind=model.info_ind,
                      system_enc=SYSTEMATIC_ENCODING,
                      code_gm=model.code_gm,
                      decoder_name="Ensemble",
                      crc_order=CONFIG.crc_order)

