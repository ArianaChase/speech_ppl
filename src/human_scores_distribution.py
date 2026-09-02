import pandas as pd
import json

# ==========================================
# PART 1: Load JSON from File & Parse
# ==========================================

json_file_path = '/home/ubuntu/speech_ppl/src/scores_enhanced.json'  # Replace with your actual file path

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index().rename(columns={'index': 'audio_filename'})
df['speaker'] = df['audio_filename']
cols_to_keep = ['audio_filename', 'speaker', 'accuracy', 'completeness', 'fluency', 'prosodic', 'age', 'gender']
df = df[cols_to_keep]

dimensions = ['accuracy', 'fluency', 'prosodic']

# ==========================================
# PART 2: Group Definitions
# ==========================================

groups = {
    'All Samples': df,
    'Female': df[df['gender'].str.lower() == 'f'],
    'Male': df[df['gender'].str.lower() == 'm'],
    '18+': df[df['age'] >= 18],
    '18-': df[df['age'] < 18],
}

# ==========================================
# PART 3: Median / IQR summary
# ==========================================

def describe_column(series):
    """Returns 'Median=.., IQR=[Q1, Q3], n=..'"""
    series = series.dropna()
    n = len(series)
    if n == 0:
        return "no data"

    median = series.median()
    q1, q3 = series.quantile([0.25, 0.75])

    note = f"  [n={n} too small to trust]" if n < 5 else ""

    return f"Median={median:.1f}, IQR=[{q1:.1f}, {q3:.1f}], n={n}{note}"


print("=== SCORE DISTRIBUTION REPORT (Median / IQR) ===\n")

for dim in dimensions:
    print(f"--- {dim.upper()} ---")
    for g_name, g_df in groups.items():
        print(f"{g_name:12s}: {describe_column(g_df[dim])}")
    print()