from flask import Flask, request, jsonify
from model_handler import ModelHandler
import os
from datetime import datetime
import json

app = Flask(__name__)

MODEL_PATH = 'models/model_v2.pkl'
SCALER_PATH = 'models/scaler_v2.pkl'

model_handler = ModelHandler(MODEL_PATH, SCALER_PATH)
print(f"Модель {model_handler.model_version} загружена")

os.makedirs('logs', exist_ok=True)

def log_request(features, prediction, probability, status, error=None):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'prediction': prediction,
        'probability': probability,
        'status': status,
        'model_version': model_handler.model_version
    }
    if error:
        log_entry['error'] = error
    with open('logs/api_logs.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'healthy': True, 
        'status': 'ok', 
        'model_version': model_handler.model_version
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'features' not in data:
            log_request(None, None, None, 'error', 'Missing features')
            return jsonify({'error': 'Missing features field'}), 400
        
        features = data['features']
        
        if len(features) != 9:
            log_request(features, None, None, 'error', f'Expected 9 features, got {len(features)}')
            return jsonify({'error': f'Expected 9 features, got {len(features)}'}), 400
        
        prediction, probability = model_handler.predict(features)
        log_request(features, prediction, probability, 'success')
        
        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 4),
            'model_version': model_handler.model_version,
            'message': 'High risk of default' if probability > 0.5 else 'Low risk of default'
        }), 200
        
    except Exception as e:
        log_request(None, None, None, 'error', str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify(model_handler.get_model_info()), 200


if __name__ == '__main__':
    print("Запуск сервера на http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)