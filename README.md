# Overview
We created a machine-learning model that predicts the chance of heart attacks using minimal user input. Since heart attacks can happen anytime, the model aims to act as an initial indicator for heart attacks. The model predicts heart attacks given four predictor variables: age, gender, systolic blood pressure, and heart rate. The variables are meant to be straightforward, so devices can quickly implement them. For example, we imagine our model working with smartwatches that measure blood pressure and heart rate and give alerts for possible heart attacks. The model's accuracy is 73% which is a reasonable estimate since variables are considered superficial. Below we elaborate on selecting variables, creating the model, and designing the model interface.  

# Dataset
We used a dataset of 303 observations for training the model, which has multiple predictor variables with one target binary variable, which shows whether the observation has a low chance or high chance of heart attack (0 or 1). For our purpose, we are interested in the independent variables easily measured by devices: age, gender, blood pressure, cholesterol level, and heart rate. However, we still need to filter them out to have the best-fit model. The dataset is derived from [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset). Here is a screenshot of the subset we use from the dataset. 

![subset](https://drive.google.com/uc?export=download&id=1R4YBVCA0BuXF8QqQo1_iUoUYM45evwOP)

# Data Preprocessing
A common pitfall in creating machine learning models is the multicollinearity between independent variables, which is the presence of correlation between independent variables. In the "Variables Selection.py" file, we have a function named "multicollinearity," which assesses the correlation coefficients between independent variables, and we set a threshold of 0.7 to consider the independent variables correlated. Hopefully, we didn't have multicollinearity between independent variables. Here is the correlation heatmap between independent variables. 

![heatmap](https://drive.google.com/uc?export=download&id=151ZE7zHxmkYiM989IIBMFt89FZCZ5ycu)

Next, we examined the correlation coefficients between the independent and target variables. Since we have six independent variables, we would need help to overfit the model. So, we excluded two variables, cholesterol level and blood sugar, because they have a very weak correlation with the target variable. We will only work with four independent variables for the final model. Here is the correlation table between each independent variable and the target variable.

![correlations](https://drive.google.com/uc?export=download&id=1QZcZJ-ApMcJOuZkd_2sTiFs-LPSYQz8M)

# Model Selection
We used the "Support Vector Classifier" machine learning algorithm for creating the model. SVC is a supervised classification model whose objective is to classify the data based on a maximal margin hyperplane build using support vectors. It is a commonly used algorithm in machine learning that provides a very accurate estimate compared to other algorithms. The final model is saved in the "model.pkl" file, which the interface will access to predict heart attacks using the user's inputs.

# Interface Desiging 
The model's interface is a web service that takes the user's inputs and implements them in the model to predict heart attacks. We used flask library to connect Python back-end development with the web front-end development. The goal of the interface is to show how the model works, and our motivation behind the model is to create a minimal accessible source to predict heart attacks for smart devices. Here is a screenshot of the interface.

![interface](https://drive.google.com/uc?export=download&id=1rWOYzI499lMrdBB2LMx4nLlysbn94shL=250x250)
<img src="https://drive.google.com/uc?export=download&id=1MacTG0k0kSGhPT2IYAKT2ipj_c1gXAbh" width="500" height="500">

