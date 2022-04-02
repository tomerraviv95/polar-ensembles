from python_code.utils.python_utils import llr_to_bits
from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.decoders.ensemble_decoder import EnsembleDecoder
from python_code.trainers.trainer import Trainer
from globals import CONFIG, DEVICE
from time import time
import numpy as np
import torch

EARLY_TERMINATION = True


class EnsembleTrainer(Trainer):
    """
    Wraps the decoder with the evaluation method
    """
    # TODO: check if need to implement different save weights
    def __init__(self):
        run_name = CONFIG.run_name
        if not(run_name):
            run_name = f"ensemble_{CONFIG.code_len}_{CONFIG.info_len}_iters{CONFIG.iteration_num}_crc{CONFIG.crc_order}_{CONFIG.ensemble_crc_dist}"
        CONFIG.set_value('run_name',run_name)
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

    def calc_loss(self, prediction, labels):
        output, not_satisfied_list = prediction
        return self.criterion(-output, labels)

    def decode(self, soft_values):
        return llr_to_bits(soft_values)

    def evaluate(self,take_crc_0=False):
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
                start = time()
                while err_count < CONFIG.test_errors:
                    ber, fer, err_indices = self.single_eval(j, take_crc_0=take_crc_0)
                    ber_total[j] += ber
                    fer_total[j] += fer
                    err_count += err_indices.shape[0]
                    snr_test_size += 1.0

                ber_total[j] /= snr_test_size
                fer_total[j] /= snr_test_size
                print(
                    f'done. time: {time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}, log-ber:{-np.log(ber_total[j])}, tot errors: {err_count}')
            return ber_total, fer_total

    def single_eval(self, j, take_crc_0=False):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output, not_satisfied_list = self.model(rx_per_snr, take_crc_0=take_crc_0)
        decoded_words = self.decode(output)

        return calculate_accuracy(decoded_words, target_per_snr, DEVICE)


if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = EnsembleTrainer()

    start = time()

    ber, fer = dec.train()

    end = time()
    print(f'################## total training time: {end-start} ##################')
