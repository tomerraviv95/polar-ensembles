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
            run_name = f"ensemble_{CONFIG.code_len}_{CONFIG.info_len}_iters{CONFIG.iteration_num}_crc{CONFIG.crc_order}_{CONFIG.ensemble_crc_dist}_decs_{CONFIG.ensemble_dec_num}"
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

    def evaluate_test(self,take_crc_0=False):
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
                    ber, fer, err_indices = self.single_eval_test(j, take_crc_0=take_crc_0)
                    ber_total[j] += ber
                    fer_total[j] += fer
                    err_count += err_indices.shape[0]
                    snr_test_size += 1.0

                ber_total[j] /= snr_test_size
                fer_total[j] /= snr_test_size
                print(
                    f'done. time: {time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}, log-ber:{-np.log(ber_total[j])}, tot errors: {err_count}')
            return ber_total, fer_total

    def single_eval_test(self, j, take_crc_0=False):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        # decode and calculate accuracy
        output, not_satisfied_list = self.model(rx_per_snr, take_crc_0=take_crc_0)
        decoded_words = self.decode(output)

        return calculate_accuracy(decoded_words, target_per_snr, DEVICE)


    def evaluate_train(self, take_crc_0=False):
        """
        Evaluation is done at every SNR until a specific number of decoding errors occur
        This ensures more stability at each point, than another method which simply simulates X points at every SNR
        :return: BER and FER vectors
        """
        dec_num = self.model.num_of_decoders
        snr_range = self.snr_range['val']
        ber_total, fer_total = np.zeros((len(snr_range), dec_num)), np.zeros((len(snr_range), dec_num))
        err_total = np.zeros(dec_num)

        with torch.no_grad():
            for j, snr in enumerate(snr_range):
                total_words_per_decoder = np.zeros(dec_num)
                print('start eval snr ' + str(snr))
                start = time()
                while min(err_total) < CONFIG.test_errors:
                    ber, fer, err_counts, words_per_decoder = self.single_eval_train(j, take_crc_0=take_crc_0)
                    ber_total[j] += ber*words_per_decoder # element wise
                    fer_total[j] += fer*words_per_decoder
                    err_total += err_counts
                    total_words_per_decoder += words_per_decoder

                ber_total[j] /= total_words_per_decoder
                fer_total[j] /= total_words_per_decoder
                print(
                    f'done. time: {time() - start}, ber: {ber_total[j]}, fer: {fer_total[j]}, log-ber:{-np.log(ber_total[j])}, tot errors: {err_total}')
            return ber_total, fer_total

    def single_eval_train(self, j, take_crc_0=False):
        # draw test data
        rx_per_snr, target_per_snr = iter(self.channel_dataset['val'][j])
        rx_per_snr = rx_per_snr.to(device=DEVICE)
        target_per_snr = target_per_snr.to(device=DEVICE)

        dec_num = self.model.num_of_decoders
        ber = np.zeros(dec_num)
        fer = np.zeros(dec_num)
        err_counts = np.zeros(dec_num)
        words_per_decoder = np.zeros(dec_num)

        # decode and calculate accuracy per decoder
        output, not_satisfied_list, dec_mask = self.model(rx_per_snr, take_crc_0=take_crc_0, get_mask=True)
        decoded_words = self.decode(output)

        for dec_idx in range(dec_num):
            decoder_id = dec_idx + 1
            idx = (dec_mask == decoder_id).flatten("F")
            words_per_decoder[dec_idx] = np.sum(idx)
            if words_per_decoder[dec_idx] == 0:
                continue # dont take into account if decoder hasn't been used
            curr_decoded_words = decoded_words[idx]
            curr_target = target_per_snr[idx]
            ber[dec_idx], fer[dec_idx], err_indices = calculate_accuracy(curr_decoded_words, curr_target, DEVICE)
            err_counts[dec_idx] = err_indices.shape[0]

        return ber, fer, err_counts, words_per_decoder


if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = EnsembleTrainer()

    start = time()

    ber, fer = dec.train()

    end = time()
    print(f'################## total training time: {end-start} ##################')
