import pandas as pd
import json
import matplotlib.pyplot as plt

# ==========================================
# PART 1: Load JSON from File & Parse
# ==========================================

json_file_path = '/home/u5504709/new_work/speech_ppl/src/scores_enhanced.json' # Replace with your actual file path

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
pretty_colors = ['#e63946', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

# Group Definitions
groups_all =    {'All Samples': df}
groups_gender = {'Female': df[df['gender'].str.lower() == 'f'], 'Male': df[df['gender'].str.lower() == 'm']}
groups_age =    {'18+': df[df['age'] >= 18], '18-': df[df['age'] < 18]}

def get_percentage(df_subset, column_name):
    if df_subset.empty: return pd.Series(0.0, index=labels)
    binned = pd.cut(df_subset[column_name], bins=bins, labels=labels)
    return (binned.value_counts(normalize=True).sort_index() * 100).round(1)

def get_average(df_subset, column_name):
    if df_subset.empty: return 0.0
    return round(df_subset[column_name].mean(), 1)


# ==========================================
# PART 3: The Text Report
# ==========================================

print("=== SCORE DISTRIBUTION REPORT ===\n")

for dim in dimensions:
    print(f"--- {dim.upper()} ---")
    
    stats = {}
    for g_name, g_df in {**groups_all, **groups_gender, **groups_age}.items():
        stats[g_name] = {'avg': get_average(g_df, dim), 'pct': get_percentage(g_df, dim)}
    
    print(f"[AVERAGES] All: {stats['All Samples']['avg']} | Females: {stats['Female']['avg']} | Males: {stats['Male']['avg']} | Adults: {stats['18+']['avg']} | Children: {stats['18-']['avg']}\n")
    
    for bracket in labels:
        print(f"{stats['All Samples']['pct'][bracket]}% of ALL SAMPLES are in the {bracket} score bracket.")
        print(f"{stats['Female']['pct'][bracket]}% of females ... {stats['Male']['pct'][bracket]}% of males.")
        print(f"{stats['18+']['pct'][bracket]}% of adults ... {stats['18-']['pct'][bracket]}% of children.\n")
    print("="*60 + "\n")


# ==========================================
# PART 4: Compact, Cinematic Chart Function
# ==========================================

def plot_compact_dashboard(groups_dict, title, filename):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(
        1, 4,
        figsize=(14, 3),
        sharex=True
    )

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, (ax, dim) in enumerate(zip(axes, dimensions)):

        temp_df = pd.DataFrame(index=labels)

        for name, subset in groups_dict.items():
            avg = get_average(subset, dim)
            temp_df[f"{name}\n({avg:.1f})"] = get_percentage(subset, dim)

        temp_df = temp_df.T

        temp_df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=pretty_colors,
            edgecolor='white',
            width=0.5
        )

        ax.set_title(dim.capitalize(), fontsize=11, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.invert_yaxis()

        ax.set_xlabel('')
        ax.set_ylabel('')

        if i != 0:
            ax.set_yticklabels([])

        ax.tick_params(axis='y', length=0)

        legend = ax.get_legend()
        if legend:
            legend.remove()

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        title='Score Brackets',
        loc='lower center',
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        title_fontsize=10
    )

    plt.tight_layout()

    plt.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.76,
        bottom=0.23,
        wspace=0.1
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"Chart saved to {filename}")

# ==========================================
# PART 5: Generate the 3 Files
# ==========================================

plot_compact_dashboard(
    groups_all,
    'Score Distributions (All Samples)',
    'all_distributions.png'
)

plot_compact_dashboard(
    groups_gender,
    'Gender Score Distributions',
    'gender_distributions.png'
)

plot_compact_dashboard(
    groups_age,
    'Age Score Distributions',
    'age_distributions.png'
)


# Display all figures on screen
plt.show()