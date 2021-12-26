from python_code.utils.python_utils import llr_to_bits
from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.data.channel_model import BPSKmodulation, AWGN
import torch
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.decoders.ensemble_decoder import EnsembleDecoder
from python_code.trainers.trainer import Trainer
from python_code.data.dataset_crc import DatasetCRC
from globals import CONFIG, DEVICE
from time import time

EARLY_TERMINATION = True
SYSTEMATIC_ENCODING = False
USE_LLR = True


class EnsembleTrainer(Trainer):
    """
    Wraps the decoder with the evaluation method
    """
    # TODO: check if need to implement different save weights
    def __init__(self):
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
        crc_dataset = DatasetCRC(load_dataset=CONFIG.load_dataset,
                                 save_dataset=not(CONFIG.load_dataset),
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

        self.channel_dataset = {}
        self.channel_dataset['train'] = crc_dataset[:,:int(0.85*CONFIG.words_per_crc_range)]
        self.channel_dataset['val'] = crc_dataset[:,int(0.85*CONFIG.words_per_crc_range):]

        self.dataloaders = {phase: torch.utils.data.DataLoader(self.channel_dataset[phase]) for phase in
                            ['train', 'val']}


    def calc_loss(self, prediction, labels):
        output, not_satisfied_list = prediction
        return self.criterion(-output, labels)

    def decode(self, soft_values):
        return llr_to_bits(soft_values)

    def single_eval(self, j):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output, not_satisfied_list = self.model(rx_per_snr)
        decoded_words = self.decode(output)

        return calculate_accuracy(decoded_words, target_per_snr, DEVICE)

if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = EnsembleTrainer()

    start = time()

    ber, fer = dec.train()

    end = time()
    print(f'################## total training time: {end-start} ##################')
