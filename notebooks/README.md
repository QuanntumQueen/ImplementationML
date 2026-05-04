## Credit Card Default ML Service

## Описание проекта

Production-like - сервис для прогнозирования дефолта по кредитным картам.
 
Проект включает:
- обучение модели RandomForestClassifier;
- Flask API с эндпоинтами `/health`, `/predict`, `/model/info`;
- контейнеризацию через Docker;
- тестирование API;
- план A/B-тестирования.

**Датасет:** Default of Credit Card Clients Dataset с UCI / Kaggle


## Структура проекта

```text
CREDIT-CARD-ML-DEPLOYMENT/
├── app/
│ ├── __ini__t.py
│ ├── api.py
│ └── model_handler.py
├── data/
│ └── UCI_Credit_Card.csv
├── docker/
│ └── Dockerfile
├── logs/
│ └── api_logs.json
├── models/
│ ├── model_v1.pkl
│ ├── model_v1.py
│ ├── model_v2.pkl
│ ├── scaler_v2.pkl
│ ├── scaler.pkl
│ └── train-1.ipynb
├── notebooks/
│ ├── ab_test_plan.md
│ ├── eda_and_model2.ipynb
│ └── README.md
├── tests/
│ └── test_api.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── healthpredict.JPG
└── requirements.txt
```

##  Модель
Используется модель **RandomForestClassifier** для задачи бинарной классификации: предсказание дефолта клиента по кредитной карте.

### Версии моделей

| Версия | Признаки | Особенности | F1-score | Recall |
|--------|----------|-------------|----------|--------|
| **v1** | 23 признака | Базовая модель | 0.4592 | 0.3519 |
| **v2** | 9 признаков( вкл.feature engineering) | `class_weight='balanced'` | 0.5310 | 0.5607 |

### Feature Engineering (новые признаки)
- `DELAY_COUNT` — сколько месяцев была просрочка у клиента
- `PAYMENT_RATIO` —  PAYMENT_RATIO — платёж / лимит -какую часть лимита клиент платит
- `BILL_TO_LIMIT` — долг / лимит -какую часть лимита должен
- `ZERO_PAYMENT_COUNT` — сколько месяцев не платил клиент

## Локальный запуск

### 1. Установка зависимостей

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Обучение модели

```bash
jupyter notebook models/train-1.ipynb
```

3. Запуск API
```bash
python app/api.py
```
Сервис будет доступен по адресу:
```text
http://localhost:5000
```
## API
### GET /health
Проверка состояния сервиса и загрузки модели.

Пример запроса:
```powershell
Invoke-RestMethod -Uri http://localhost:5000/health -Method GET | ConvertTo-Json
```
Пример ответа:
```json
{
  "healthy": true,
  "model_version": "v2",
  "status": "ok"
}
```

### POST /predict

Принимает JSON с признаками клиента и возвращает:
`prediction` — предсказанный класс (0/1);
`probability` — вероятность дефолта;
`model_version` — версия модели;
`message` — текстовое описание риска.

Пример запроса:
```powershell
Invoke-RestMethod `
  -Uri http://localhost:5000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"features": [0, 0, 0, 0, 0.05, 0.2, 0, 35, 140000]}' `
  ConvertTo-Json
```
Пример ответа:
```json
{
  "prediction": 0,
  "probability": 0.1472,
  "model_version": "v2",
  "message": "Low risk of default"
}
```
### GET /model/info
Информация о загруженной модели.

Пример запроса:
```powershell
Invoke-RestMethod -Uri http://localhost:5000/model/info -Method GET | ConvertTo-Json
Пример ответа:
```json
{
  "features_count": 9,
  "is_loaded": true,
  "model_type": "RandomForestClassifier",
  "model_version": "v2"
}
```
## Тесты
Запуск тестов API:
```bash
python tests/test_api.py
```
Результат: 5/5 тестов пройдено

## Docker
### Сборка образа
```bash
docker build -f docker/Dockerfile -t credit_default_model:latest .
```
### Запуск контейнера
```bash
docker run --rm -p 5000:5000 credit_default_model:latest
```
## Docker Compose
Запуск:
```bash
docker-compose up --build
```
Остановка:
```bash
docker-compose down
```
##  Docker Hub
Ссылка на опубликованный Docker-образ:
Docker Hub: https://hub.docker.com/r/3067094mu/credit_default_model
sha256:419f7d348067a9cc296a80f395bfe263772caf2196c33f331dd6565005014ebc

##  Скачивание и запуск:

```bash
docker pull 3067094mu/credit_default_model:latest
docker run -d -p 5000:5000 3067094mu/credit_default_model:latest
```

## Демонстрация
Скриншот работы API:
(https://github.com/QuanntumQueen/ImplementationML/blob/main/healthpredict.JPG)

## Результаты:
GET /health → сервис работает, модель v2 загружена
POST /predict → получен корректный прогноз

### Архитектура: монолит vs микросервисы
В рамках данного учебного проекта выбран монолитный подход.
Причины:
- минимальная сложность для MVP;
- быстрее разработка и деплой;
- меньше операционных накладных расходов;
- достаточно для одного ML use-case.

## Логирование, мониторинг и MLOps-концепты
### RabbitMQ (концепт)
В production-сценарии RabbitMQ можно использовать для:
- асинхронного batch scoring;
- retraining jobs;
- логирования и доставки событий в очередь.

### Логирование
API-запросы логируются в logs/api_logs.json в JSON-формате:

```json
{
  "timestamp": "2026-05-04T10:30:00",
  "features": [0, 0, 0, 0, 0.05, 0.2, 0, 35, 140000],
  "prediction": 0,
  "probability": 0.1472,
  "status": "success",
  "model_version": "v2"
}
```
В production такие логи могут централизованно собираться через ELK / OpenSearch / Grafana stack.

### DVC (концепт)
- DVC используется для контроля версий данных и ML-артефактов, а также для воспроизводимости пайплайна.

### MLflow (концепт)
- MLflow используется для трекинга экспериментов, хранения метрик, параметров и артефактов моделей.

### ONNX-ML, uWSGI и NGINX (концепты)
ONNX-ML
- Модель scikit-learn можно преобразовать в ONNX-формат для ускоренного инференса и более удобного кросс-платформенного развёртывания.

uWSGI + NGINX
В production среде:
- uWSGI / Gunicorn выступает как WSGI-сервер для Python-приложения;
- тNGINX работает как reverse proxy, распределяет запросы, обрабатывает TLS, статику и балансировку нагрузки.

### Бизнес-метрики
Помимо технических метрик (F1-score, Precision, Recall) используются бизнес-метрики:
 - ожидаемое снижение потерь от дефолтов;
 - доля одобренных заявок при фиксированном уровне риска.

