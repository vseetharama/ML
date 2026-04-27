import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load data
df = fetch_california_housing(as_frame=True).frame

# Numerical columns
cols = df.select_dtypes(include=np.number).columns

# -------- Histograms --------
plt.figure(figsize=(15,10))
for i in range(len(cols)):
    plt.subplot(3,3,i+1)
    sns.histplot(df[cols[i]], kde=True, bins=45,color='blue')
    plt.title(cols[i])
plt.tight_layout()
plt.show()

# -------- Boxplots --------
plt.figure(figsize=(15,10))
for i in range(len(cols)):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=df[cols[i]],color='orange')
    plt.title(cols[i])
plt.tight_layout()
plt.show()

# -------- Outliers --------
print("Outliers count:")

outliers_summary = {}

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    outliers_summary[col] = count

print(outliers_summary)

# -------- Describe --------
print(df.describe())
