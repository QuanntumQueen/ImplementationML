import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import joblib
import os

def main():
    print("Обучение модели прогнозирования дефолта")

    # Загрузка данных
    file_path = r'C:\Users\Anna\Desktop\MAGA2\implementationML\session\credit-card-ml-deployment\data\UCI_Credit_Card.csv'
    df = pd.read_csv(file_path)
    print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    # Подготовка данных
    y = df['default.payment.next.month']
    X = df.drop(['ID', 'default.payment.next.month'], axis=1)
    print(f"   Признаков: {X.shape[1]}")
    print(f"   Дефолтов: {y.mean()*100:.1f}%")

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   train выборка: {X_train.shape[0]} строк")
    print(f"   test выборка: {X_test.shape[0]} строк")

    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели
    model = RandomForestClassifier(random_state=42, n_estimators=90, max_depth=10)
    model.fit(X_train_scaled, y_train)
  

    # Оценка модели
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Результаты:")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # Сохранение модели
    joblib.dump(model, 'model_v1.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("model_v1.pkl сохранён в папке models/")
    print("scaler.pkl сохранён в папке models/")


if __name__ == '__main__':
    main()