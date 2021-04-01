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

    # calculate train loss
    def calc_loss(self, prediction, labels):
        output_list, not_satisfied_list = prediction
        if CONFIG.multiloss:
            total_loss = 0
            for output, not_satisfied in zip(output_list, not_satisfied_list):
                if type(output) == int:
                    break
                total_loss += self.criterion(output, labels[not_satisfied])
            # total_loss+=self.criterion(output_list[-1], labels)
            return total_loss
        else:
            return self.criterion(output_list[-1], labels)


if __name__ == "__main__":
    # load config and run evaluation of decoder
    dec = PolarFGTrainer()
    ber, fer = dec.evaluate()
