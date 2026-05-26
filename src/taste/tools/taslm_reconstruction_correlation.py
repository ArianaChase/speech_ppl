import pandas as pd
import csv
import scipy.stats

output_csv = "/home/u5504709/new_work/speech_ppl/work/outputs/taslm_reconstruction_001"
output_csv_df = pd.read_csv(output_csv)
x = output_csv_df["Raw MCD"].values


def calc_correlation(x, dim):
    if (dim == "accuracy"):
        y = output_csv_df["Human Annotation (Accuracy)"]
    elif (dim == "fluency"):
        y = output_csv_df["Human Annotation (Fluency)"]
    elif (dim == "prosodic"):
        y = output_csv_df["Human Annotation (Prosody)"]
    elif (dim == "completeness"):
        y = output_csv_df["Human Annotation (Completeness)"]
    else:
        print(f"Invalid dimension")
        return
    
    print(f"=== Correlation for dimension {dim} ===")
    print("Correlation x len: ", len(x))
    print("Correlation y len: ", len(y))
    print(f"Correlation value is: {scipy.stats.pearsonr(x, y)}")

calc_correlation(x, "accuracy")
calc_correlation(x, "fluency")
calc_correlation(x, "prosodic")
calc_correlation(x, "completeness")
