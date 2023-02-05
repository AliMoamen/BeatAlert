# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle 

# Load the data from the local dataset
df = pd.read_csv("heart.csv")

# Subset the data to keep only relevant columns
subset = df[["age" , "sex" ,"trtbps" , "thalachh" , "output"]]
subset.head()

# Scale the feature data
sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(data = X)

# Split the data into feature and target variables
X = subset.iloc[:, [0 , 1,  2 , 3]].values
Y = subset["output"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = subset["output"] )

# Train the SVC model with a linear kernel
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)

# Save the trained model
pickle.dump(svc, open('model.pkl','wb'))

# Load the saved model
model = pickle.load(open('model.pkl','rb'))

# Make a prediction using the loaded model
print("Prediction: ", model.predict(np.array([[55,	1,	132,	132]])))

# Print the model's test accuracy score
print("Model Score: ", svc.score(X_test, Y_test)*100 , "%")
