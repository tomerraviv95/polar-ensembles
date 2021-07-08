from python_code.utils.python_utils import llr_to_bits
from python_code.decoders.fg_decoder import FGDecoder
from python_code.trainers.trainer import Trainer
from globals import CONFIG, DEVICE

EARLY_TERMINATION = True


class PolarFGTrainer(Trainer):
    """
    Wraps the decoder with the evaluation method - pretty straightforward
    """

    def __init__(self):
        super().__init__()

    def load_model(self):
        self.model = FGDecoder(code_len=CONFIG.code_len,
                               info_len=CONFIG.info_len,
                               design_snr=CONFIG.design_SNR,
                               iteration_num=CONFIG.iteration_num,
                               clipping_val=CONFIG.clipping_val,
                               device=DEVICE)
        self.decoder_name = 'FG'

    # calculate train loss
    def calc_loss(self, prediction, labels):
        output_list, not_satisfied_list = prediction
        return self.criterion(output_list[-1], labels)

    def decode(self, soft_values):
        return llr_to_bits(soft_values)


if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = PolarFGTrainer()
    ber, fer = dec.train()
