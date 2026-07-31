import json




for model in ["TWIST-1.3B", "GSLM"]:
    for pool in ["mean", "max", "std"]:
        PATH = f"/home/u5504709/new_work/speech_ppl/src/gslm/tools/result_dicts/{model}_word_{pool}_norm.json"

        print(f"Printing {model} stats at {pool} pooling method")

        with open(PATH, "r") as f:
            word_dict = json.load(f)

        for key, value in word_dict.items():
            print(f"Bucket {key}, {value['count']} samples")
            print(f"{value['mean']} loss on average")
            print(f"{value['std']} standard deviation")


