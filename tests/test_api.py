import requests
import json

BASE_URL = "http://localhost:5000"

# Пример признаков для модели v2 (9 чисел)
SAMPLE_FEATURES = [
    0, 0, 0,           # PAY_0, PAY_2, PAY_3 (нет просрочек)
    0,                 # DELAY_COUNT (0 просрочек)
    0.05,              # PAYMENT_RATIO (платит 5% от лимита)
    0.2,               # BILL_TO_LIMIT (долг 20% от лимита)
    0,                 # ZERO_PAYMENT_COUNT (всегда платит)
    35,                # AGE (35 лет)
    140000             # LIMIT_BAL (лимит 140k)
]

def test_health():
    print("тест 1: GET /health")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200

def test_predict():
    print("тест 2: POST /predict")
  
    payload = {"features": SAMPLE_FEATURES}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Prediction: {data['prediction']}")
        print(f"Probability: {data['probability']}")
        print(f"Message: {data['message']}")
    else:
        print(f"Error: {response.text}")
    print()
    return response.status_code == 200

def test_invalid_features():
    print("тест 3: POST /predict (некорректные данные)")
  
    payload = {"features": [1, 2, 3]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()
    return response.status_code == 400

def test_model_info():
    print("тест 4: GET /model/info")
   
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    return response.status_code == 200

def test_missing_features():
  
    print("тест 5: POST /predict (отсутствует поле features)")
  
    
    payload = {"wrong_field": [1, 2, 3]}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()
    return response.status_code == 400

if __name__ == "__main__":
    
    print(f"Базовый URL: {BASE_URL}")
    
    tests = [
        ("GET /health", test_health),
        ("POST /predict", test_predict),
        ("POST /predict (некорректные данные)", test_invalid_features),
        ("GET /model/info", test_model_info),
        ("POST /predict (отсутствует features)", test_missing_features)
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
   
# Результат тестов  
    passed = 0
    for name, result in results:
        status = "пройден" if result else "не пройден"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"Всего пройдено: {passed}/{len(tests)}")