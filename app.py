from flask import Flask, request, jsonify
from model import MLModel
import numpy as np

app = Flask(__name__)
model = MLModel()

# Contoh pelatihan model dengan data dummy
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([2, 3, 4, 5])
model.train(X_train, y_train)

@app.route('/')
def home():
    return 'Welcome to the ML Model API. Use /predict to get predictions.'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json.get('data')
        if data is None:
            return jsonify({'error': 'No data provided'}), 400

        try:
            # Pastikan data yang diterima dalam bentuk array dua dimensi
            data = np.array(data).reshape(1, -1)
            prediction = model.predict(data)
            return jsonify({'prediction': prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True, port=5000)
