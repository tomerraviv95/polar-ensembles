from dir_definitions import WEIGHTS_DIR, CONFIG_PATH
from shutil import copyfile
from python_code.utils.python_utils import llr_to_bits
from python_code.utils.evaluation_criterion import calculate_accuracy
from python_code.trainers.fg_trainer import PolarFGTrainer
from python_code.decoders.ensemble_decoder import EnsembleDecoder
from python_code.trainers.trainer import Trainer
from globals import CONFIG, DEVICE
from time import time
import numpy as np
import torch
import os

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

    def train(self):
        self.optimization_setup()
        self.loss_setup()
        snr_range = self.snr_range['train']
        self.evaluate_train()
        dec_num = self.model.num_of_decoders
        ber_total, fer_total, best_ber = np.ones(dec_num), np.ones(dec_num), np.ones(dec_num)
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
                ber_total, fer_total = self.evaluate_train()
                # extract relevant ber, either scalar or last value in list
                if ber_total.shape[0] != 1:
                    raise ValueError('Must run training with single eval SNR!!!')

                for dec_idx in range(dec_num):
                    decoder_id = dec_idx + 1
                    ber = ber_total[0,dec_idx]
                    fer = fer_total[0,dec_idx]

                    # save weights if model is improved compared to best ber
                    if ber < best_ber[dec_idx]:
                        self.save_weights(epoch, decoder_id)
                        best_ber[dec_idx] = ber

        return ber_total, fer_total

    def setup_save_dir(self):
        self.weights_dir = os.path.join(WEIGHTS_DIR, CONFIG.run_name)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            # save config in output dir
            copyfile(CONFIG_PATH, os.path.join(self.weights_dir, "config.yaml"))
        for dec_id,dec in enumerate(self.model.decoders):
            if dec_id == 0:
                continue
            folder = os.path.join(self.weights_dir, str(dec_id))
            if not os.path.exists(folder):
                os.makedirs(folder)


    def save_weights(self, epoch, decoder_id):
        torch.save({'model_state_dict': self.model.decoders[decoder_id].state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch},
                   os.path.join(self.weights_dir, f'{decoder_id}\\epoch_{epoch}.pt'))

    def load_weights(self):
        """
        Loads detector's weights defined by the [snr,gamma] from checkpoint, if exists
        """
        for dec_id,dec in enumerate(self.model.decoders):
            folder = os.path.join(self.weights_dir, str(dec_id))
            if os.path.isdir(folder):
                files = os.listdir(folder)
                names = []
                for file in files:
                    if file.startswith("epoch_"):
                        names.append(int(file.split('.')[0].split('_')[1]))
                names.sort()
                print(f'loading model from epoch {names[-1]}')
                checkpoint = torch.load(os.path.join(folder, 'epoch_' + str(names[-1]) + '.pt'))
                try:
                    dec.load_state_dict(checkpoint['model_state_dict'])
                except Exception:
                    raise ValueError("Wrong run directory!!!")
            else:
                print(f'No such dir!!! starting from scratch')

if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = EnsembleTrainer()

    start = time()

    ber, fer = dec.train()

    end = time()
    print(f'################## total training time: {end-start} ##################')
