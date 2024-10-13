import os
import pickle
import bz2
import logging
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load models with error handling
try:
    pickle_in = bz2.BZ2File(os.path.join('model', 'classification.pkl'), 'rb')
    model_C = pickle.load(pickle_in)
    logging.info("Classification model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading classification model: {e}")
    model_C = None  # Set to None to handle later in the predict function

try:
    R_pickle_in = bz2.BZ2File(os.path.join('model', 'regression.pkl'), 'rb')
    model_R = pickle.load(R_pickle_in)
    logging.info("Regression model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading regression model: {e}")
    model_R = None  # Set to None to handle later in the predict function

# Initialize the scaler
scaler = StandardScaler()

# Create a DataFrame for fitting the scaler
df = pd.DataFrame({'Temperature': [0], 'Ws': [0], 'FFMC': [0], 'DMC': [0], 'ISI': [0], 'FWI': [0], 'Classes': [0]})
X = df.drop(['FWI', 'Classes'], axis=1)

# Fit the scaler to the dummy data
scaler.fit(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    
    # Transform the features using the scaler
    final_features = scaler.transform(final_features)
    
    # Check if the model is loaded
    if model_C is None:
        return render_template('index.html', prediction_text1="Model not loaded. Please check the server logs.")

    # Make a prediction
    output = model_C.predict(final_features)[0]
    text = 'Forest is Safe!' if output == 0 else 'Forest is in Danger!'
    return render_template('index.html', prediction_text1=f"{text} --- Chance of Fire is {output}")

@app.route('/predictR', methods=['POST'])
def predictR():
    data = [float(x) for x in request.form.values()]
    data = [np.array(data)]
    
    # Transform the features using the scaler
    data = scaler.transform(data)
    
    # Check if the model is loaded
    if model_R is None:
        return render_template('index.html', prediction_text2="Model not loaded. Please check the server logs.")

    # Make a prediction
    output = model_R.predict(data)[0]
    return render_template('index.html', prediction_text2=f"Fuel Moisture Code index is {output:.4f}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
