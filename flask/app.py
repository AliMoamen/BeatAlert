# Importing required libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler


# Creating Flask app
app = Flask(__name__)

# Loading saved machine learning model
model = pickle.load(open('model.pkl', 'rb'))

# Reading the heart disease dataset
df = pd.read_csv('heart.csv')

# Selecting the relevant columns
subset = df[["age" , "sex" ,"trtbps" , "thalachh" , "output"]]

# Extracting the feature values as a numpy array
dataset_X = subset.iloc[:,[0, 1, 2, 3]].values

# Importing the MinMaxScaler from scikit-learn library
from sklearn.preprocessing import MinMaxScaler

# Initializing the StandardScaler object
sc = StandardScaler()

# Scaling the feature values to a range of 0 to 1
dataset_scaled = sc.fit_transform(dataset_X)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict',methods=['POST'])
def predict():

    # Extracting the float values from the form inputs
    float_features = [float(x) for x in request.form.values()]

    # Converting the list of float values to a numpy array
    final_features = np.array(float_features)

    # Storing the feature values as a list
    information = list(final_features)

    # Converting the gender value (0 or 1) to its string representation
    if information[1] == 0:
        information[1] = "Female"
    elif information[1] == 1:
        information[1] = "Male"

    # Using the machine learning model to make a prediction
    prediction = model.predict(np.array([list(final_features)]))

    # Converting the prediction (0 or 1) to its string representation
    if prediction[0] == 1:
        pred = "High Chance of Heart Attack"
    elif prediction[0] == 0:
        pred = "Low Chance of Heart Attack"
    output = pred

    # Rendering the index.html template and passing the prediction result
    return render_template('index.html', prediction_text='{}'.format(output),  data = "{}".format("Age: " + str(int(information[0])) + " years"+ ", Gender: " + information[1] + ", Systolic Blood Pressure: " + str(int(information[2])) + " mm Hg"+ ", Maximum Heart Rate: " + str(int(information[3]) )+ " bpm"))

# Main function
if __name__ == "__main__":
    # Running the Flask app in debug mode
    app.run(debug=True)
