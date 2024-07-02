import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

score_labels = {
    1: 'Very bad (1)',
    2: 'Bad (2)',
    3: 'Fair (3)',
    4: 'Good (4)',
    5: 'Excellent (5)'
}

# Colors for the ratings
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3']

# Define metrics and methods
metrics = ['Smoothness', 'Diversity', 'Beat Accuracy', 'Overall']
methods = ['DanceFusion', 'EDGE']

# Initialize dictionaries to count scores for each method and metric
score_counts = {method: {metric: {i: 0 for i in range(1, 6)} for metric in metrics} for method in methods}

# Load the CSV file
data_path = r'visualize_result\\user_study\\In-the-wild.csv'
df = pd.read_csv(data_path)

# Calculate the total number of pairs of videos
num_pairs = 6

# Iterate through the rows and count scores
for index, row in df.iterrows():
    for pair in range(num_pairs):
        for metric_idx, metric in enumerate(metrics):
            for method_idx, method in enumerate(methods):
                score_index_start = 4 + (pair * 8) + (method_idx * 4) + metric_idx
                score = int(row[score_index_start])
                score_counts[method][metric][score] += 1

# Create horizontal stacked bar plots
fig, axs = plt.subplots(1, len(metrics), figsize=(25, 2), sharey=True)

# Function to plot horizontal stacked bar chart
def plot_horizontal_stacked_bar(score_counts, metric, ax):
    labels = range(1, 6)
    data = {method: [score_counts[method][metric][label] for label in labels] for method in methods}
    
    bottom = np.zeros(len(methods))  # Initialize bottom to zero for all bars
    bar_width = 0.35  # Set a narrower width for the bars

    for i, label in enumerate(labels):
        values = [score_counts[method][metric][label] for method in methods]
        ax.barh([0, 0.5], values, left=bottom, color=colors[i], edgecolor='white', height=0.4, align='center')
        bottom += values
    
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_yticks([0, 0.5])
    ax.set_yticklabels(methods)
    ax.set_xlabel(metric)
    ax.xaxis.set_label_position('bottom')

# Plot for each metric in a horizontally stacked bar fashion
for j, metric in enumerate(metrics):
    plot_horizontal_stacked_bar(score_counts, metric, axs[j])

# Custom bar notation
legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
fig.legend(
    handles=legend_handles,
    labels=[score_labels[i] for i in range(1, 6)],
    loc='upper center', 
    ncol=5,
    bbox_to_anchor=(0.5, 1.0),
    fontsize=12,
    frameon=False,
)

# Adjust layout slightly for better fit
plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.75, left=0.075) # Add space at the top for the legend
plt.show()