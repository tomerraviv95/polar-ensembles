from utils.parse_config import parse_config_file
from evaluators.polar_FG_evaluator import PolarFGEvaluator
import matplotlib.pyplot as plt

code_len_vec = [32, 64, 128]
info_len_vec = [16, 32, 64]
colors = ['brown', 'blue', 'green']
markers = ['*', '^', 'x']

configuration = parse_config_file()

for i, (code_len, info_len) in enumerate(zip(code_len_vec, info_len_vec)):
    configuration["code_len"] = code_len
    configuration["info_len"] = info_len
    dec: PolarFGEvaluator = PolarFGEvaluator(configuration)
    ber, fer = dec.evaluate()

    plt.plot(dec.val_SNRs, fer, label=f"Polar Code ({code_len},{info_len})",
             color=colors[i], marker=markers[i])

plt.title('FER Comparison of Different Polar Codes')
plt.yscale('log')
plt.xlabel("Eb/N0(dB)")
plt.ylabel("FER")
plt.legend(loc='upper right')
plt.grid(which='both', linestyle='--')
plt.xlim((dec.val_SNRs[0] - 0.5, dec.val_SNRs[-1] + 0.5))

plt.show()
