import pandas as pd
import json
import matplotlib.pyplot as plt

# ==========================================
# PART 1: Load JSON from File & Parse
# ==========================================

json_file_path = 'data.json' # Replace with your actual file path

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame.from_dict(data, orient='index')
df = df.reset_index().rename(columns={'index': 'audio_filename'})
df['speaker'] = df['audio_filename'] 
cols_to_keep = ['audio_filename', 'speaker', 'accuracy', 'completeness', 'fluency', 'prosodic', 'age', 'gender']
df = df[cols_to_keep]


# ==========================================
# PART 2: Define Categories & Math Functions
# ==========================================

bins = [-1, 2, 4, 6, 8, 10]
labels = ['0-2', '3-4', '5-6', '7-8', '9-10']
dimensions = ['accuracy', 'completeness', 'fluency', 'prosodic']

# A modern, pretty color gradient for the 5 score brackets (Red -> Orange -> Yellow -> Teal -> Dark Blue)
pretty_colors = ['#e63946', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

groups = {
    'Female': df[df['gender'].str.lower() == 'f'],
    'Male': df[df['gender'].str.lower() == 'm'],
    '18+': df[df['age'] >= 18],
    '18-': df[df['age'] < 18]
}

def get_percentage(df_subset, column_name):
    if df_subset.empty: return pd.Series(0.0, index=labels)
    binned = pd.cut(df_subset[column_name], bins=bins, labels=labels)
    return (binned.value_counts(normalize=True).sort_index() * 100).round(1)

def get_average(df_subset, column_name):
    if df_subset.empty: return 0.0
    return round(df_subset[column_name].mean(), 1)


# ==========================================
# PART 3: Generate the Text Report
# ==========================================

print("=== SCORE DISTRIBUTION REPORT ===\n")

for dim in dimensions:
    print(f"--- {dim.upper()} ---")
    
    avg_female = get_average(groups['Female'], dim)
    avg_male = get_average(groups['Male'], dim)
    avg_adult = get_average(groups['18+'], dim)
    avg_child = get_average(groups['18-'], dim)
    
    pct_female = get_percentage(groups['Female'], dim)
    pct_male = get_percentage(groups['Male'], dim)
    pct_adult = get_percentage(groups['18+'], dim)
    pct_child = get_percentage(groups['18-'], dim)
    
    print(f"[AVERAGE SCORES] Adults: {avg_adult}/10 | Children: {avg_child}/10 | Females: {avg_female}/10 | Males: {avg_male}/10\n")
    
    for bracket in labels:
        print(f"{pct_adult[bracket]}% of adults are in the {bracket} score bracket while "
              f"{pct_child[bracket]}% of children are in the {bracket} score bracket.")
        print(f"{pct_female[bracket]}% of females are in the {bracket} score bracket while "
              f"{pct_male[bracket]}% of males are in the {bracket} score bracket.\n")
    print("="*80 + "\n")


# ==========================================
# PART 4: Generate File 1 (GENDER CHART)
# ==========================================

plt.style.use('seaborn-v0_8-whitegrid')

# 4 rows, 1 column for a vertical stack of horizontal bar charts
fig_gen, axes_gen = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
fig_gen.suptitle('Gender Score Distributions (100% Stacked)', fontsize=18, fontweight='bold')

for i, dim in enumerate(dimensions):
    avg_female = get_average(groups['Female'], dim)
    avg_male = get_average(groups['Male'], dim)
    
    gender_df = pd.DataFrame(index=labels)
    gender_df[f"Female\n(Avg: {avg_female})"] = get_percentage(groups['Female'], dim)
    gender_df[f"Male\n(Avg: {avg_male})"] = get_percentage(groups['Male'], dim)
    gender_df = gender_df.T 
        
    ax = axes_gen[i]
    # kind='barh' makes the bars horizontal
    gender_df.plot(kind='barh', stacked=True, ax=ax, color=pretty_colors, edgecolor='white', width=0.6)
    
    ax.set_title(f'{dim.capitalize()}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Percentage (%)', fontsize=11) 
    ax.set_xlim(0, 100) 
    ax.invert_yaxis() # Puts 'Female' on top, 'Male' on bottom (reads more naturally)
    
    # Only put the legend on the very top chart
    if i == 0: 
        ax.legend(title='Score Brackets', bbox_to_anchor=(1.02, 1), loc='upper left')
    else: 
        ax.get_legend().remove()

fig_gen.tight_layout()
fig_gen.subplots_adjust(top=0.92, right=0.85)

filename_gen = 'gender_distributions.png'
fig_gen.savefig(filename_gen, dpi=300, bbox_inches='tight')
print(f"Chart successfully saved to {filename_gen}!")


# ==========================================
# PART 5: Generate File 2 (AGE CHART)
# ==========================================

fig_age, axes_age = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
fig_age.suptitle('Age Score Distributions (100% Stacked)', fontsize=18, fontweight='bold')

for i, dim in enumerate(dimensions):
    avg_adult = get_average(groups['18+'], dim)
    avg_child = get_average(groups['18-'], dim)
    
    age_df = pd.DataFrame(index=labels)
    age_df[f"18+\n(Avg: {avg_adult})"] = get_percentage(groups['18+'], dim)
    age_df[f"18-\n(Avg: {avg_child})"] = get_percentage(groups['18-'], dim)
    age_df = age_df.T 
        
    ax = axes_age[i]
    age_df.plot(kind='barh', stacked=True, ax=ax, color=pretty_colors, edgecolor='white', width=0.6)
    
    ax.set_title(f'{dim.capitalize()}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Percentage (%)', fontsize=11) 
    ax.set_xlim(0, 100) 
    ax.invert_yaxis() # Puts '18+' on top, '18-' on bottom
    
    if i == 0: 
        ax.legend(title='Score Brackets', bbox_to_anchor=(1.02, 1), loc='upper left')
    else: 
        ax.get_legend().remove()

fig_age.tight_layout()
fig_age.subplots_adjust(top=0.92, right=0.85)

filename_age = 'age_distributions.png'
fig_age.savefig(filename_age, dpi=300, bbox_inches='tight')
print(f"Chart successfully saved to {filename_age}!\n")

# Show both plots on the screen
plt.show()