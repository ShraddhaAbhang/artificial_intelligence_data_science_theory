# To get started with the fundamentals of data science, you can break down the key components into various code examples. 
# Here, I'll give you a basic overview of how you might approach these components using Python. 
# This will include some essential libraries and concepts.

### 1. Data Collection

# To work with data, you first need to collect it. 
# Here's a simple example of how to load a dataset using Python’s `pandas` library:

import pandas as pd

# Load a dataset from a CSV file
df = pd.read_csv('data.csv')

# Display the first few rows of the dataset
print(df.head())

### 2. Data Cleaning

# Data cleaning is crucial in data science to ensure that your data is accurate and ready for analysis. 
# Here’s a basic example of handling missing values and duplicates:

# Drop rows with missing values
df_cleaned = df.dropna()

# Drop duplicate rows
df_cleaned = df_cleaned.drop_duplicates()

# Fill missing values with a specific value, for example, the mean
df_filled = df.fillna(df.mean())

### 3. Data Exploration

# Exploring the data involves understanding its structure, statistics, and patterns. 
# Here’s how you can get basic statistics and visualizations:

# Get basic statistics
print(df.describe())

# Get the correlation matrix
print(df.corr())

import matplotlib.pyplot as plt

# Plot a histogram of a specific column
df['column_name'].hist()
plt.show()

# Plot a scatter plot between two columns
df.plot.scatter(x='column1', y='column2')
plt.show()

### 4. Data Analysis

# Data analysis often involves applying statistical or machine learning techniques. 
# Here's a basic example of fitting a linear regression model using `scikit-learn`:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features and target variable
X = df[['feature1', 'feature2']]  # Example features
y = df['target']  # Example target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

### 5. Data Visualization

# Effective data visualization helps in understanding complex data insights. 
# Here’s a basic example using `seaborn`:

import seaborn as sns

# Create a pairplot to visualize relationships between features
sns.pairplot(df[['feature1', 'feature2', 'target']])
plt.show()

# Create a heatmap of the correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

### 6. Basic Machine Learning

# Here’s a quick example of a classification task using a decision tree:

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a sample dataset
data = load_iris()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# These examples provide a basic overview of some fundamental components of data science. 
# As you dive deeper, you'll explore more advanced topics and techniques, but understanding these basics will give you a solid foundation to build upon.

