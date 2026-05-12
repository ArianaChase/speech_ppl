import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import argparse
import json
from operator import itemgetter
import csv

#  def dim_correlation(x, dim):
#     if (dim == "accuracy"):
#         y = output_csv_df["Human Annotation (Accuracy)"]
#     elif (dim == "fluency"):
#         y = output_csv_df["Human Annotation (Fluency)"]
#     elif (dim == "prosodic"):
#         y = output_csv_df["Human Annotation (Prosody)"]
#     elif (dim == "completeness"):
#         y = output_csv_df["Human Annotation (Completeness)"]
#     else:
#         print(f"Invalid dimension")
#         return
    
#     print(f"=== Correlation for dimension {dim} ===")
#     print("Correlation x len: ", len(x))
#     print("Correlation y len: ", len(y))
#     print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")
    

# def calc_correlation(file, labels, mode="combined"):
#     loss_df = pd.read_csv(file)
#     y = labels

#     losses = []

#     if (mode == "utt_token"):
#         losses = loss_df["utt_token_loss"].astype(float).values
#         x = -np.array(losses)

#         dim_correlation(losses, "accuracy")
#         dim_correlation(losses, "fluency")
#         dim_correlation(losses, "prosodic")
#         dim_correlation(losses, "completeness")

#     elif (mode == "utt_flow"):
#         losses = loss_df["utt_flow_loss"].astype(float).values
#         x = -np.array(losses)

#         dim_correlation(losses, "accuracy")
#         dim_correlation(losses, "fluency")
#         dim_correlation(losses, "prosodic")
#         dim_correlation(losses, "completeness")
#     elif (mode == "combined"):
#         losses = loss_df["weighted_combined_score"].astype(float).values
#         x = -np.array(losses)

#         dim_correlation(losses, "accuracy")
#         dim_correlation(losses, "fluency")
#         dim_correlation(losses, "prosodic")
#         dim_correlation(losses, "completeness")
#     else:
#         print("Mode not valid.")




def parse_human_annotations(filename):
    human_scores = []
    with open(filename) as json_data:
        data = json.load(json_data)
        for audio_file in data:
            value = data[audio_file]
            human_scores.append({
                "filename" : audio_file,
                "accuracy" : value["accuracy"],
                "fluency" : value["fluency"],
                "prosodic" : value["prosodic"],
                "completeness" : value["completeness"]
            })
    return human_scores

def losses_to_dict(loss_file, labels_list):

    with open(loss_file, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    loss_objects = []
    

    for loss_obj in data:
        filename = loss_obj["id"][12:]
        speaker = loss_obj["id"][7:11]


        if (any(filename == x["filename"] for x in labels_list)):    
            loss_objects.append({
                "filename" : filename,
                "raw_token_losses" : loss_obj['raw_token_losses'],
                "raw_flow_losses" : loss_obj['raw_flow_losses'],
                "utt_token_loss" : loss_obj['utt_token_loss'],
                "utt_flow_loss" : loss_obj['utt_flow_loss']
            })
    
    print("Length of list: ", len(loss_objects))
    print("Length of labels: ", len(labels_list))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--loss_file", help="Path to losses", default=None)
    parser.add_argument("--labels_dir", help="Path to labels", default=None)

    args = parser.parse_args()

    loss_file = args.loss_file

    # get labels to compare to
    score_labels = args.labels_dir
    human_scores = parse_human_annotations(score_labels)
    human_scores = sorted(human_scores, key=itemgetter("filename"))

    losses_to_dict(loss_file, human_scores)





