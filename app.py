from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  

model = joblib.load('logistic_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        prediction_result = prediction[0].item()  
        return jsonify({'prediction': int(prediction_result)})  
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
