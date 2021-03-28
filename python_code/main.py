from python_code.utils.parse_config import parse_config_file
from python_code.evaluators import PolarFGEvaluator

if __name__=="__main__":
    #load config and run evaluation of decoder
    configuration = parse_config_file()
    dec = PolarFGEvaluator(configuration)
    ber, fer = dec.evaluate()