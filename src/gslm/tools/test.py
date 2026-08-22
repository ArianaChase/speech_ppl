import json
import numpy as np

PATH = f"/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts/COSYVOICE_word_max_norm.json"


with open(PATH, "r") as f:
    phone_dict = json.load(f)

for key, value in phone_dict.items():
    losses = [x for x in value['losses'] if x == x]
    value['losses'] = losses
    value['mean'] = np.mean(value['losses'])
    value['std'] = np.std(value['losses'])
    value['count'] = len(value['losses'])

with open(PATH, "w") as f:
    json.dump(phone_dict, f)

