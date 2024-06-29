import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the CSV file
csv_file_path = r'visualize_result\\user_study\\In-the-wild.csv'
df = pd.read_csv(csv_file_path, header=None)

# Extract the relevant columns and set column names
user_info_df = df.iloc[:, 0:4]
user_info_df.columns = ["Gender", "Age", "AI Knowledge", "Choreography Knowledge"]

# Ensure categorical columns are string type for consistent plotting
categorical_columns = ["Gender", "Age", "AI Knowledge", "Choreography Knowledge"]
user_info_df[categorical_columns] = user_info_df[categorical_columns].astype(str)

# Create subplots for each category in 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)

# Titles for each plot
titles = ['Gender', 'Age', 'AI Knowledge', 'Choreography Knowledge']
colors = sns.color_palette("pastel")

for idx, ax in enumerate(axes.flat):
    category = categorical_columns[idx]
    category_counts = user_info_df[category].value_counts()
    
    # Create pie chart with decorative features
    wedges, texts, autotexts = ax.pie(
        category_counts, 
        labels=category_counts.index, 
        autopct='%1.1f%%', 
        colors=sns.color_palette("pastel", len(category_counts)), # Gradient color palette
        startangle=140, 
        explode=[0.1 if i == 0 else 0 for i in range(len(category_counts))]  # Explode the largest slice
    )
    
    ax.set_title(titles[idx], size=15, weight='bold')

    for text in texts:
        text.set_fontsize(12)  # Increase label size
        text.set_color("navy")

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_color("black")

# Add super title to the figure
plt.suptitle("User Information", size=20, weight='bold', y=0.98)
plt.show()