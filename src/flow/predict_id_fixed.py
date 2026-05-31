import os
from pathlib import Path
import json
from operator import itemgetter


dataset = "/home/u5504709/new_work/speech_ppl/speechocean762/WAVE"

def parse_human_annotations(filename):
    human_scores = []
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            if value["age"] < 18:
                human_scores.append({
                    "filename" : os.path.basename(audio_file),
                    "accuracy" : value["accuracy"],
                    "fluency" : value["fluency"],
                    "prosodic" : value["prosodic"],
                    "completeness" : value["completeness"]
                })
    return human_scores

def create_predict_id(dataset, human_scores):
    score_map = {obj["filename"]: obj for obj in human_scores}

    out_path = "/home/u5504709/new_work/speech_ppl/src/predict_id_agednot18.txt"

    total = 0

    with open(out_path, "w") as f:
        for speaker in os.listdir(dataset):
            if speaker[7:11] == "1076":
                continue

            speaker_path = os.path.join(dataset, speaker)

            for file in os.listdir(speaker_path):
                stem = Path(file).stem

                if stem not in score_map:
                    continue

                f.write(f"{speaker}/{stem}\n")
                total += 1

    print("Total files processed: ", total)

    return total
    

score_labels = "/home/u5504709/new_work/speech_ppl/src/scores_enhanced.json"
human_scores = parse_human_annotations(score_labels)
human_scores = sorted(human_scores, key=itemgetter("filename"))

create_predict_id(dataset, human_scores)

