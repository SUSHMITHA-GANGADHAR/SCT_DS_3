import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset
data = pd.read_csv(r'C:/Users/DELL/Downloads/task1.csv', skiprows=4)

# Process data and calculate mean population by year
years = [str(year) for year in range(1970, 2010)]
mean_population_by_year = data[years].mean()

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Bar chart with 'husl' palette for different color shades
sns.barplot(x=mean_population_by_year.index, y=mean_population_by_year,hue=None ,palette="husl", ax=axes[0])
axes[0].set_title('Average Population by Year (Bar Chart)')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Population')
axes[0].grid(False)

# Histogram with a single color
sns.histplot(mean_population_by_year, kde=False, color='orange', edgecolor='black', ax=axes[1])
axes[1].set_title('Distribution of Average Population (Histogram)')
axes[1].set_xlabel('Average Population')
axes[1].set_ylabel('Frequency')
axes[1].grid(True)

# Adjust subplot spacing
fig.subplots_adjust(hspace=1)
plt.tight_layout()
plt.show()
