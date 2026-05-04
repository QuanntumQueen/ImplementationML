import os
import joblib
import numpy as np

class ModelHandler:
    def __init__(self, model_path='models/model_v2.pkl', scaler_path='models/scaler_v2.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.model_version = "v2"
        self.is_loaded = True
    
    # Предсказание для одного клиента
    def predict(self, features):
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        prediction = int(self.model.predict(features_scaled)[0])
        probability = float(self.model.predict_proba(features_scaled)[0][1])
           
        return prediction, probability
    
    # Возвращает информацию о модели
    def get_model_info(self):
        return {
            'model_version': self.model_version,
            'model_type': str(type(self.model).__name__),
            'features_count': 9,
            'is_loaded': self.is_loaded
        }