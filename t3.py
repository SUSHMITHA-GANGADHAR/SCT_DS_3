import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import tree
import numpy as np

# Load the dataset
dataset_path = 'C:/Users/DELL/Downloads/bank+marketing/bank-additional/bank-additional/bank-additional.csv'  # Update with your file path
df = pd.read_csv(dataset_path, delimiter=';')
print("Data loaded successfully.")
print(df.head())

# Identify the target column (assuming it's the last column)
target_column = df.columns[-1]
print(f"Target column identified as: {target_column}")

# Label encoding for categorical variables, excluding the target column
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != target_column:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode the target variable
df[target_column] = df[target_column].map({'yes': 1, 'no': 0})

# Split the data into train and test sets
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance plot
importance = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

# Plotting bar plots for categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
n_features = len(categorical_columns)

fig, axes = plt.subplots(nrows=int(np.ceil(n_features / 3)), ncols=3, figsize=(16, 12))
for i, col in enumerate(categorical_columns):
    sns.countplot(x=col, data=df, ax=axes.flat[i])
    axes.flat[i].set_title(col)
plt.tight_layout()
plt.show()

# Plotting box plots for numerical features
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(nrows=int(np.ceil(len(numeric_columns) / 3)), ncols=3, figsize=(16, 12))
for i, col in enumerate(numeric_columns):
    sns.boxplot(x=target_column, y=col, data=df, ax=axes.flat[i])
    axes.flat[i].set_title(col)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Full decision tree visualization
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()
