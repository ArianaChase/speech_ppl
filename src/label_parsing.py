import json
import csv

def parse_accuracy_scores(filename):
    accuracy_scores = []
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            accuracy_scores.append(value["accuracy"])

    return accuracy_scores


score_labels = "/Users/hermitcrab/speech_ppl/speechocean762/resource/scores.json"

print(len(parse_accuracy_scores(score_labels)))
         

