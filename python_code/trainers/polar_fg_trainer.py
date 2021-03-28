from python_code.decoders.polar_fg_decoder import PolarFGDecoder
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
        self.model = PolarFGDecoder(code_len=CONFIG.code_len,
                                    info_len=CONFIG.info_len,
                                    design_SNR=CONFIG.design_SNR,
                                    crc=CONFIG.crc,
                                    iteration_num=CONFIG.iteration_num,
                                    clipping_val=CONFIG.clipping_val,
                                    early_termination=EARLY_TERMINATION,
                                    device=DEVICE)


if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = PolarFGTrainer()
    ber, fer = dec.train()
