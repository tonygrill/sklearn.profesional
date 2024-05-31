import joblib
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

# Ruta de bienvenida
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the prediction API!"

# Ruta de predicción
@app.route('/predict', methods=['GET'])
def predict():
    print("Received request for /predict")
    X_test = np.array([7.074656748,6.911343313,1.546259284,1.419920564,0.774286628,0.505740523,0.392578781,0.135638788,2.218113422])
    prediction = model.predict(X_test.reshape(1, -1))
    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    model = joblib.load('./models/best_model.pkl')
    print("Model loaded successfully")
    app.run(port=8999)
