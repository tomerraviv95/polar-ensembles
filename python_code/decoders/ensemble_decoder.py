from python_code.utils.python_utils import llr_to_bits
from python_code.decoders.decoder import Decoder
from python_code.decoders.fg_decoder import FGDecoder
from python_code.codes.crc import crc
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
        self.InitSelectorByCrc(ensemble_crc_dist)
        self.decoders_mask = [0] * (self.num_of_decoders + 1)
        self._build_model()

    def _build_model(self):
        # define layers
        decoders = [0] * (self.num_of_decoders + 1)
        for i in range(self.num_of_decoders+1):
            decoders[i] = FGDecoder(self.code_len, self.info_len, self.design_snr, self.clipping_val, self.iteration_num, self.device)
        self.decoders = torch.nn.ModuleList(decoders)

    def forward(self, rx: torch.Tensor):
        """
        compute forward pass in the network
        :param rx: [batch_size,N]

        decode CRC then use it to cancel the NN that do not designated to it
        """
        output = torch.zeros((rx.size(0), self.info_len), device=rx.device)
        for i in range(len(self.decoders_mask)):
            self.decoders_mask[i] = torch.zeros((rx.size(0), rx.size(0)), device=rx.device)
        output_t, not_satisfied = self.decoders[0](rx)
        decoded_words = llr_to_bits(output_t[-1])
        pred_crc = crc.crc_check(decoded_words, CONFIG.crc_order)
        crc_vals = crc.crc2int(pred_crc)
        for idx, val in enumerate(crc_vals):
            dec_id = self.selector(val)
            self.decoders_mask[dec_id][idx,idx] = 1 # this will mask words from decoders with crc not in range


        for id,dec in enumerate(self.decoders):
            if id != 0: # decoder number 0 already decoded for crc value
                output_t, not_satisfied = dec(rx)
            output += torch.matmul(self.decoders_mask[id],output_t[-1]) # taking the last iteration
            # print(f'dec number {id} got {torch.tensor.sum(self.decoders_mask[id])} / {rx.size(0)} words to decode')

        return output, not_satisfied

    def InitSelectorByCrc(self, crc_dist):
        vals = 2**self.crc_order
        if crc_dist == 'uniform':
            self.selector = lambda x : 0 if x == 0 else 1 + int((x-1)*self.num_of_decoders/vals)
        else:
            print(f'crc dist choice: {crc_dist} is not yet implemented')
            raise ValueError