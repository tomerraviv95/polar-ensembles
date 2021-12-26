from dir_definitions import WEIGHTS_DIR, CONFIG_PATH, DATA_DIR
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

SYSTEMATIC_ENCODING = False


class DatasetCRC(ChannelModelDataset):
    ''' generate dataset with uniform crc_range distribution over each SNR'''
    def __init__(self, load_dataset, save_dataset, words_per_crc, model_name, code_len, info_len, code_type,
                 clipping_val, decoder_name, use_llr=True,
                 modulation=BPSKmodulation,
                 channel=AWGN, batch_size=None,
                 snr_range=None, zero_word_only=True,
                 random=None, wordRandom=None,
                 crc_order=0,
                 **code_params):

        super().__init__(self, code_len, info_len, code_type,
                  clipping_val, decoder_name, use_llr=use_llr,
                  modulation=modulation,
                  channel=channel, batch_size=batch_size,
                  snr_range=snr_range, zero_word_only=zero_word_only,
                  random=random, wordRandom=wordRandom,
                  crc_order=crc_order,
                  **code_params)

        self.load_dataset = load_dataset
        self.save_dataset = save_dataset
        self.words_per_crc = words_per_crc
        self.words_per_crc = []
        self.data = []
        self.model_name = model_name
        self.load_model()

        if load_dataset:
            self.load_data()
        else:
            self.generate_data()

    def load_model(self):
        if self.model_name == "Ensemble":
            self.model = EnsembleDecoder(code_len=self.code_len,
                                         info_len=self.info_len,
                                         design_snr=CONFIG.design_SNR,
                                         iteration_num=CONFIG.iteration_num,
                                         clipping_val=self.clipping_val,
                                         device=DEVICE,
                                         crc_order=self.crc_order,
                                         num_of_decoders=CONFIG.ensemble_dec_num,
                                         ensemble_crc_dist=CONFIG.ensemble_crc_dist)
            self.decoder_name = 'Ensemble'

        elif self.model_name == "FG":
            print(f"model {self.model_name} is not supported")
            raise ValueError
            # self.model = FGDecoder(code_len=self.code_len,
            #                        info_len=self.info_len,
            #                        design_snr=CONFIG.design_SNR,
            #                        iteration_num=CONFIG.iteration_num,
            #                        clipping_val=self.clipping_val,
            #                        device=DEVICE)
            # self.decoder_name = 'FG'

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

        for j, snr in enumerate(self.snr_range):
            while sum(words_per_crc_counter[j]) < 4*self.words_per_crc:
                # draw test data
                rx_per_snr, target_per_snr = iter(self.channel_dataset[j])
                rx_per_snr = rx_per_snr.to(device=DEVICE)
                target_per_snr = target_per_snr.to(device=DEVICE)

                # decode and calculate accuracy
                output, not_satisfied_list = self.model(rx_per_snr)
                decoded_words = self.decode(output)
                pred_crc = crc.crc_check(decoded_words, self.crc_order)

                for val in pred_crc:
                    # check if need to save this crc val or already have enough



        if self.save_dataset:
            self.save_data()

    def load_data(self):
        pass

    def save_data(self):
        pass

    def __getitem__(self, item):
        return self.data[item]


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
                                                           decoder_name=self.decoder_name,
                                                           crc_order=CONFIG.crc_order)
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

    def evaluate_crc(self):
        '''
        Evaluate the crc value for every snr
        '''
        snr_range = self.snr_range['val']
        pred_crc_distribution = np.zeros((len(snr_range), 2**CONFIG.crc_order))
        actual_crc_distribution = np.zeros((len(snr_range), 2**CONFIG.crc_order))
        crc2int = lambda crc : int("".join(str(int(x)) for x in crc),2)
        for j, snr in enumerate(snr_range):
            rx_per_snr_tot, target_per_snr_tot = iter(self.channel_dataset['val'][j])
            for i in range(10): # divide to batch
                ptr = int(CONFIG.val_batch_size/10)
                rx_per_snr = rx_per_snr_tot[i*ptr:(i+1)*ptr]
                target_per_snr = target_per_snr_tot[i*ptr:(i+1)*ptr]
                rx_per_snr = rx_per_snr.to(device=DEVICE)
                target_per_snr = target_per_snr.to(device=DEVICE)
                output_list, not_satisfied_list = self.model(rx_per_snr)
                decoded_words = self.decode(output_list[-1])
                actual_crc = crc.crc_check(target_per_snr, CONFIG.crc_order)
                pred_crc = crc.crc_check(decoded_words, CONFIG.crc_order)
                for w in range(np.shape(actual_crc)[0]):
                    actual_val = crc2int(actual_crc[w])
                    pred_val = crc2int(pred_crc[w])
                    actual_crc_distribution[j,actual_val] += 1
                    pred_crc_distribution[j,pred_val] += 1

                print(f'done {int(100*(i+10*j)/(10*len(snr_range)))}%')

        return actual_crc_distribution, pred_crc_distribution



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

