from flask import Flask, render_template, request, jsonify
from sklearn.naive_bayes import GaussianNB
import numpy as np

app = Flask(__name__)

# Dữ liệu huấn luyện mẫu
X_train = np.array([
    [30, 70, 1, 15], [32, 65, 1, 12], [27, 90, 0, 20], [25, 85, 0, 18],
    [35, 50, 1, 10], [38, 45, 1, 8], [22, 96, 0, 22], [28, 80, 0, 17],
    [33, 60, 1, 10], [36, 40, 1, 9], [29, 78, 0, 19], [31, 63, 1, 13],
    [39, 42, 1, 7], [21, 99, 0, 24], [34, 55, 1, 11]
])
y_train = np.array([
    1, 1, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 0, 1, 0
])

model = GaussianNB()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        sunny = int(data['sunny'])
        wind = float(data['wind'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Vui lòng nhập đầy đủ và đúng định dạng các trường!'}), 400

    X = np.array([[temp, humidity, sunny, wind]])
    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return jsonify({
        'prediction': prediction,
        'probability': {
            'rain': float(proba[1]),
            'no_rain': float(proba[0])
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)