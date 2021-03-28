from python_code.trainers.polar_fg_trainer import PolarFGTrainer
import matplotlib.pyplot as plt
from globals import CONFIG
import numpy as np

code_len_vec = [32, 64, 128]
info_len_vec = [16, 32, 64]
all_runs_params = [{'code_len': 32, 'info_len': 16},
                   {'code_len': 64, 'info_len': 32},
                   {'code_len': 128, 'info_len': 64}]
colors = ['brown', 'blue', 'green']
markers = ['*', '^', 'x']

val_SNRs = np.linspace(CONFIG.val_SNR_start, CONFIG.val_SNR_end, num=CONFIG.val_num_SNR)

for i, run_params in enumerate(all_runs_params):
    # set all parameters based on dict
    for k, v in run_params.items():
        CONFIG.set_value(k, v)
    dec = PolarFGTrainer()
    ber, fer = dec.evaluate()
    plt.plot(val_SNRs, fer, label=f"Polar Code ({run_params['code_len']},{run_params['info_len']})",
             color=colors[i], marker=markers[i])

plt.title('FER Comparison of Different Polar Codes')
plt.yscale('log')
plt.xlabel("Eb/N0(dB)")
plt.ylabel("FER")
plt.legend(loc='upper right')
plt.grid(which='both', linestyle='--')
plt.xlim((val_SNRs[0] - 0.5, val_SNRs[-1] + 0.5))

plt.show()
