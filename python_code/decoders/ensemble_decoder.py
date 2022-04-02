from python_code.utils.python_utils import llr_to_bits
from python_code.decoders.decoder import Decoder
from python_code.decoders.fg_decoder import FGDecoder
from python_code.codes.crc import crc
import numpy as np
from globals import CONFIG


import torch

EARLY_TERMINATION_EVAL = True


class EnsembleDecoder(Decoder):
    """
    hold 1 FG decoder (dec id 0) for CRC evaluation and N WFG decoders (dec id 1-N), each with different targeted CRC range decoding
    """

    def __init__(self, code_len, info_len, design_snr, clipping_val, iteration_num, device, crc_order, num_of_decoders, ensemble_crc_dist):
        super().__init__(code_len, info_len, design_snr, clipping_val, iteration_num, device)

        self.num_of_decoders = num_of_decoders
        self.crc_order = crc_order
        self.crc_dist = ensemble_crc_dist
        self.InitSelectorByCrc()
        self.decoders_mask = [0] * (self.num_of_decoders + 1)
        self._build_model()
        self.generateCRCpassrateDict()
        self.keep_crc_passrate = False

    def _build_model(self):
        # define layers
        decoders = [0] * (self.num_of_decoders + 1)
        for i in range(self.num_of_decoders+1):
            decoders[i] = FGDecoder(self.code_len, self.info_len, self.design_snr, self.clipping_val, self.iteration_num, self.device)
        decoders[0].requires_grad_(requires_grad=False) # crc selector should not be trained
        self.decoders = torch.nn.ModuleList(decoders)

    def generateCRCpassrateDict(self):
        '''save here for each decoder how many successful words it decoded for each crc range'''
        self.crc_passrate = {}
        for dec_id,dec in enumerate(self.decoders):
            self.crc_passrate[dec_id] = np.zeros((self.num_of_decoders+1,3))

    def trackCRCpassrate(self, crc_vals_dict, dec_mask):
        '''save the crc passrate for each decoder. array of [pass count of designated and curr, pass count of designated and not curr , total]'''
        if not self.keep_crc_passrate:
            return
        for dec_id,dec in enumerate(self.decoders):
            if dec_id == 0:#irrelevant
                continue
            crc_vals = crc_vals_dict[dec_id]
            for i,val in enumerate(crc_vals):
                crc_bin = dec_mask[i]
                if crc_bin == 0: # don't care about the first BP
                    continue
                designated_dec_pass = (crc_vals_dict[crc_bin][i] == 0)
                if (val == 0) and designated_dec_pass:
                    self.crc_passrate[dec_id][crc_bin][0] += 1 # curr decoder and designated one success
                elif (val == 0) and not(designated_dec_pass):
                    self.crc_passrate[dec_id][crc_bin][1] += 1 # keep track if the designated decoder succeed in this word and curr failed
                self.crc_passrate[dec_id][crc_bin][2] += 1

    def forward(self, rx: torch.Tensor, take_crc_0=False):
        """
        compute forward pass in the network
        :param rx: [batch_size,N]

        decode CRC then use it to cancel the NN that do not designated to it
        """
        batch_size = rx.size(0)
        output = torch.zeros((batch_size, self.info_len), device=rx.device)
        output_t, not_satisfied = self.decoders[0](rx)
        decoded_words = llr_to_bits(output_t[-1])
        pred_crc = torch.Tensor(crc.crc_check(decoded_words, self.crc_order))
        np_dec_mask = self.getDecodersMask(pred_crc)
        crc_vals_dict = {}

        if take_crc_0:
            best_output = torch.full((batch_size, self.info_len),-1, dtype=torch.float, device=rx.device)
            for dec_id,dec in enumerate(self.decoders):
                if dec_id != 0: # already decoded
                    output_t, not_satisfied = dec(rx)
                words = output_t[-1]
                decoded_words = llr_to_bits(words)
                crc_vals = crc.crc2int(crc.crc_check(decoded_words, self.crc_order))
                idx1 = (np_dec_mask == dec_id).flatten("F")
                idx2 = (crc_vals == 0).flatten("F")
                best_output[idx1] = words[idx1]
                best_output[idx2] = words[idx2]
                crc_vals_dict[dec_id] = crc_vals.copy() # TODO check if mutated
            self.trackCRCpassrate(crc_vals_dict=crc_vals_dict, dec_mask=np_dec_mask)
            output = best_output

        else:
            for dec_id,dec in enumerate(self.decoders):
                if dec_id != 0: # already decoded
                    output_t, not_satisfied = dec(rx)
                words = output_t[-1]
                idx = (np_dec_mask == dec_id).flatten("F")
                output[idx] = words[idx]
                if self.keep_crc_passrate:
                    crc_vals = crc.crc2int(crc.crc_check(llr_to_bits(words), self.crc_order))
                    crc_vals_dict[dec_id] = crc_vals.copy()
            self.trackCRCpassrate(crc_vals_dict=crc_vals_dict, dec_mask=np_dec_mask)
        return output, not_satisfied


    def InitSelectorByCrc(self):
        if self.crc_dist == 'uniform':
            vals = 2**self.crc_order
            self.selector = lambda x : 0 if x == 0 else (1 + int((x-1)*self.num_of_decoders/vals))
        elif self.crc_dist == 'sum':
            vals = self.crc_order + 1
            self.selector = lambda x : 0 if x == 0 else (1 + (x-1)//(vals/self.num_of_decoders))
        else:
            print(f'crc dist choice: {self.crc_dist} is not yet implemented')
            raise ValueError

    def getDecodersMask(self, crc_bin):
        whole = (self.num_of_decoders+1)//2
        res = (self.num_of_decoders+1)%2
        mask_size = whole + res
        if self.crc_dist == 'uniform':
            msb_bits = self.num_of_decoders//2 + self.num_of_decoders%2
            mask = crc.addBin(crc_bin[:,0:msb_bits],1)
            zeros = (torch.sum(crc_bin, dim=1) == 0).view(-1)
            mask[zeros] = torch.zeros((1,mask_size))
            mask = crc.crc2int(mask)
        elif self.crc_dist == 'sum':
            max_val = self.crc_order + 1
            vals = crc.sumBits(crc_bin)
            one = torch.ones(np.shape(vals))
            mask = one + (vals-one)//(max_val/self.num_of_decoders)
            zeros = (torch.sum(crc_bin, dim=1) == 0).view(-1)
            mask = mask.cpu().detach().numpy().astype(dtype=int)


            mask = np.remainder(vals,4) + 1 # use remainder of sum
            mask[zeros] = 0
            mask = mask.cpu().detach().numpy().astype(dtype=int)
            # v = vals.cpu().detach().numpy()
            # a=2
        else:
            print(f'crc dist choice: {self.crc_dist} is not yet implemented')
            raise ValueError
        return mask