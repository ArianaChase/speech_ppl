import json

PATH = f"/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts/Flow-SLM_word_mean_flow_norm.json"

with open(PATH, "r") as f:
    word_dict = json.load(f)

for key, value in word_dict.items():
    print(f"Bucket {key}, {value['count']} samples")
    print(f"{value['mean']} loss on average")
    print(f"{value['std']} standard deviation")