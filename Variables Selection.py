# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("heart.csv")
# Subsetting the dataframe to consider relevant columns
subset = df[["age" , "sex" ,"trtbps" , "chol" , "fbs" , "thalachh" , "output"]]
# Features
X = subset.drop(["output"] , axis = 1)
# Target
Y = subset["output"]
# Splitting the data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = subset["output"] )

# Plotting heatmap to visualize the correlation matrix
corr = X_train.corr()
sns.heatmap(corr, annot=True)
plt.show()

# Function to identify columns with multicollinearity
def correlation(data, threshold):
    col_corr = set()  
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]                  
                col_corr.add(colname)
    if(col_corr == set()):
        return "No Multicollinearity"
    return col_corr  

# Checking for columns with high correlation in features
corr_features = correlation(X_train, 0.7) 
# Checking the correlation between features and target
corr_target = X.corrwith(Y)

# Printing the results
print(corr_features)
print(corr_target)
