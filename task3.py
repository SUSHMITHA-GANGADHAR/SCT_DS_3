# Required Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Load the dataset from CSV
df = pd.read_csv('D:/bank-additional.csv')  # Replace 'bank-additional.csv' with your file path if needed

# Inspect the first few rows to understand the structure of your data
print(df.head())

# Print the column names to identify the target column
print("Columns in dataset:", df.columns)

# Assuming the target column is 'y' (replace this with the correct column name)
df['y'] = df['y'].map({'yes': 1, 'no': 0})  # Replace 'y' with your actual target column if different

# Drop rows with missing values if necessary
df.dropna(inplace=True)

# Split data into features (X) and target (y)
X = df.drop(columns=['y'])  # Features are all columns except the target
y = df['y']  # Target column

# Convert categorical variables to dummy/indicator variables (one-hot encoding)
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))

# Select only numeric columns for correlation matrix
df_numeric = df.select_dtypes(include=['float64', 'int64'])

sns.heatmap(df_numeric.corr(), annot=True, cmap='rainbow', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Decision Tree Visualization (Limited Depth)
clf_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_limited.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(clf_limited, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree Visualization (Limited Depth)')
plt.tight_layout()
plt.show()

# Decision Tree Visualization (Full Depth)
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(clf_full, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title('Full Decision Tree Visualization')
plt.tight_layout()
plt.show()

# Predictions and Evaluation
y_pred = clf_full.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Feature Importance
feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': clf_full.feature_importances_})
feature_importances = feature_importances.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances, palette='Blues_d')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

# Bar plots for Categorical Features
categorical_columns = df.select_dtypes(include=['object']).columns
num_cols = 3
num_rows = len(categorical_columns) // num_cols + 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
for i, feature in enumerate(categorical_columns):
    ax = axes[i // num_cols, i % num_cols]
    sns.countplot(x=feature, data=df, ax=ax)
    ax.set_title(f'Bar plot of {feature}', fontsize=10)
plt.tight_layout()
plt.show()

# Box plots for Numerical Features
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
num_cols = 3
num_rows = len(numerical_columns) // num_cols + 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
for i, feature in enumerate(numerical_columns):
    ax = axes[i // num_cols, i % num_cols]
    sns.boxplot(x=feature, data=df, ax=ax)
    ax.set_title(f'Box plot of {feature}', fontsize=10)
plt.tight_layout()
plt.show()
