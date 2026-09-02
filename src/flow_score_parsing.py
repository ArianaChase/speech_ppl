import pandas as pd
import re

# ==========================================
# PART 1: Load CSV
# ==========================================

csv_path = '/home/ubuntu/speech_ppl/src/Pronunciation Evaluation Results - new_flow.csv'  # adjust if needed

df = pd.read_csv(csv_path)

# ==========================================
# PART 2: Extract dimension + age group
# ==========================================

# Category looks like "Accuracy-Likelihood_Correlation" -> we just want "Accuracy"
df['dimension'] = df['Category'].str.split('-').str[0]

# ID looks like "flow1bext_likelihood_aged18_3" or "...agednot18_7"
# -> pull out the age group tag, ignore the run index at the end
def extract_age_group(id_str):
    match = re.search(r'(agednot18|aged18)_\d+$', id_str)
    return match.group(1) if match else None

df['age_group'] = df['ID'].apply(extract_age_group)

# keep only the 3 dimensions requested, and only rows tagged aged18 / agednot18
dimensions = ['Accuracy', 'Fluency', 'Prosody']
mask = df['dimension'].isin(dimensions) & df['age_group'].notna()
sub = df[mask].copy()

# ==========================================
# PART 3: Average over the 10 indexed runs
# ==========================================
# groups by Model x dimension x age_group, averaging 'Correlation value'
# (each group should contain 10 rows -> the 10 indexed runs)

summary = (
    sub.groupby(['Model', 'dimension', 'age_group'])['Correlation value']
    .agg(['mean', 'count'])
    .reset_index()
)

# sanity check: flag any group that doesn't have exactly 10 runs
off = summary[summary['count'] != 10]
if not off.empty:
    print("!! WARNING: these groups don't have exactly 10 runs -- check the data:")
    print(off.to_string(index=False))
    print()

# ==========================================
# PART 4: Reshape into a concise table
# Rows = Model, Columns = (dimension, age_group)
# ==========================================

table = summary.pivot_table(
    index='Model',
    columns=['dimension', 'age_group'],
    values='mean'
)

# order columns: Accuracy/Fluency/Prosody, each aged18 then agednot18
table = table.reindex(
    columns=pd.MultiIndex.from_product([dimensions, ['aged18', 'agednot18']]),
)
table = table.round(2)

# order rows: 1Bext, 1B, 270M, each acoustic then semantic
model_order = [
    'Flow-SLM-1Bext_acoustic', 'Flow-SLM-1Bext_semantic',
    'Flow-SLM-1B_acoustic', 'Flow-SLM-1B_semantic',
    'Flow-SLM-270M_acoustic', 'Flow-SLM-270M_semantic',
]
table = table.reindex(model_order)

pd.set_option('display.width', 120)
print("=== AVG CORRELATION BY MODEL x DIMENSION x AGE GROUP (over 10 runs) ===\n")
print(table.to_string())