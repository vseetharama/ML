import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing 

# Load dataset
data = fetch_california_housing(as_frame=True)
housing_df = data.frame

# Select numerical features
numerical_features = housing_df.select_dtypes(include=[np.number])

print(numerical_features.head())

# Histograms
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features.columns):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data=housing_df[feature], kde=True, bins=45, color='blue')
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(data=housing_df[feature], color='orange')
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()

# Outlier detection
print("Description of Outliers")
outliers_summary = {}

for feature in numerical_features.columns:
    Q1 = housing_df[feature].quantile(0.25)
    Q3 = housing_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    outliers = housing_df[(housing_df[feature] < lb) | (housing_df[feature] > ub)]
    outliers_summary[feature] = len(outliers)

print(outliers_summary)

# Dataset description
print(housing_df.describe())
