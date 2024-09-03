import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/DELL/Downloads/titanic/train.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Data Cleaning
# 1. Handle missing values
data = data.dropna(subset=['Age', 'Fare', 'Survived'])  # Drop rows with missing values in these columns

# 2. Remove duplicate rows
data = data.drop_duplicates()

# 3. Convert data types if necessary
# Example: Convert 'Survived' column to categorical
data['Survived'] = data['Survived'].astype('category')

# Display basic statistics
print("\nBasic statistics of the dataset:")
print(data.describe(include='all'))

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival Count
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Fare Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Fare'], kde=True, bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Correlation Heatmap
# Filter numeric columns only
numeric_data = data.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(data[['Age', 'Fare', 'Survived']])
#plt.title('pairplot')
#plt.figure(figsize=(12, 8))
#plt.title()
plt.show()
