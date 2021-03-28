from utils.parse_config import parse_config_file
from evaluators.polar_FG_evaluator import PolarFGEvaluator

if __name__=="__main__":
    #load config and run evaluation of decoder
    configuration = parse_config_file()
    dec = PolarFGEvaluator(configuration)
    ber, fer = dec.evaluate()