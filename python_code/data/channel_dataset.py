from python_code.data.channel_model import BPSKmodulation, AWGN
from python_code.polar_codes import encoding
from torch.utils.data import Dataset
import concurrent.futures
import collections
import numpy as np
import torch


class ChannelModelDataset(Dataset):
    def __init__(self, code_len, info_len, code_type, use_llr=True, modulation=BPSKmodulation, channel=AWGN,
                 batch_size=None, snr_range=None, zero_word_only=True,
                 random=None, wordRandom=None, clipping_val=15, crc_len=0, **code_params):
        self.code_len = code_len
        self.info_len = info_len
        self.crc_len = crc_len
        self.code_type = code_type
        self.random = random if random else np.random.RandomState()
        self.wordRandom = wordRandom if wordRandom else np.random.RandomState()
        self.use_llr = use_llr
        self.modulation = modulation
        self.channel = channel
        self.batch_size = batch_size
        self.snr_range = snr_range
        self.zero_word_only = zero_word_only
        self.clipping_val = clipping_val

        if code_type == 'Polar':
            try:
                self.encoding = lambda u: encoding.encode(target=u, crc_gm=code_params['crc_gm'].cpu().numpy(),
                                                          code_len=self.code_len,
                                                          info_ind=code_params['info_ind'].cpu().numpy(),
                                                          crc_ind=code_params['crc_ind'].cpu().numpy(),
                                                          system_enc=code_params['system_enc'],
                                                          factor_graph=code_params['factor_graph'].cpu().numpy())
            except:
                raise Exception('Polar code must have information bit set and flag for systematic encoding')

        else:
            raise Exception(f'code type {code_type} not implemented')

    def get_snr_data(self, snr, database=[]):
        """
        creates the data
        the while loop is here to ensure minimal batch size
        (It is used with batch-filtering such as very noisy words....
        I removed the filtering here)
        """
        rate = float(self.info_len + self.crc_len) / self.code_len
        rx = np.empty((0, self.code_len))
        tx = np.empty((0, self.code_len))
        u = np.empty((0, self.info_len))
        while len(rx) < self.batch_size:
            if self.zero_word_only:
                x = np.zeros((self.batch_size, self.code_len))
                target = np.zeros((self.batch_size, self.info_len))
                current_tx = np.ones((self.batch_size, self.code_len))
            else:
                # generate word
                target = self.wordRandom.randint(0, 2, size=(self.batch_size, self.info_len))
                # encoding
                x = self.encoding(target)
                # modulation
                current_tx = self.modulation(x)
            # add channel noise
            current_rx = self.channel(tx=current_tx, SNR=snr, R=rate, use_llr=self.use_llr, random=self.random)

            rx = np.vstack((rx, current_rx))
            tx = np.vstack((tx, x))
            u = np.vstack((u, target))

        database.append((rx[:self.batch_size], tx[:self.batch_size], u[:self.batch_size]))

    def __getitem__(self, snr_ind):
        """
        Get data for specific SNR
        """
        if not isinstance(self.snr_range, collections.Iterable):
            self.snr_range = [self.snr_range]
        if not isinstance(snr_ind, slice):
            snr_ind = [snr_ind]
        database = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            {executor.submit(self.get_snr_data, snr, database) for snr in self.snr_range[snr_ind]}
        received, sent, target = (np.concatenate(tensors) for tensors in zip(*database))

        # convert to tensor
        received = torch.tensor(received).float().view(-1, self.code_len)
        target = torch.tensor(target).float().view(-1, self.info_len)
        return received, target

    def __len__(self):
        return self.batch_size * len(self.snr_range)
