import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "https://loan-eligibility-prediction-website.vercel.app",
    "http://localhost:3000" 
])

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
FEATURES = ['Engine_size', 'Horsepower', 'Wheelbase', 'Width',
            'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency']

model = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        return False
    model = joblib.load(MODEL_PATH)
    return True


model_loaded = load_model()


@app.route('/', methods=['GET'])
def home():
    """Home endpoint - Menampilkan informasi API Prediksi Harga Mobil"""
    return jsonify({
        "message": "Car Price Prediction API",
        "version": "1.0",
        "status": "online" if model_loaded else "model_not_found",
        "endpoints": {
            "/": "GET - Informasi API (Halaman ini)",
            "/health": "GET - Cek status kesehatan API dan loading model",
            "/predict": "POST - Prediksi harga mobil berdasarkan spesifikasi teknis"
        },
        "model_details": {
            "task": "Regression",
            "target": "Price (in thousands USD)",
            "features_count": len(FEATURES),
            "required_features": FEATURES
        },
        "note": "Gunakan endpoint /predict dengan metode POST dan body JSON untuk mendapatkan estimasi harga."
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'features': FEATURES
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    data = request.get_json()
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({
            'status': 'error',
            'message': f'Missing features: {missing}',
            'required_features': FEATURES
        }), 400
    try:
        input_dict = {}
        for f in FEATURES:
            val = data[f]
            if val is None or (isinstance(val, str) and val.strip() == ''):
                raise ValueError(f'Feature "{f}" cannot be empty')
            input_dict[f] = float(val)
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        predicted_price = round(prediction * 1000, 2)
        return jsonify({
            'status': 'success',
            'predicted_price': predicted_price,
            'currency': 'USD',
            'input_features': input_dict
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    if not model_loaded:
        print('WARNING: Model file not found. Run train_model.py first.')
        print(f'Expected model at: {MODEL_PATH}')
    print(f'Features: {FEATURES}')
    app.run(debug=True, host='0.0.0.0', port=8080)
