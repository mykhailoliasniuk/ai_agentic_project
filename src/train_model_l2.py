import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from pathlib import Path


def train_and_evaluate_v2(data_path: str, optimal_threshold: float = 0.35):
    path = Path(data_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {data_path} не знайдено. Спочатку запустіть build_dataset_v2.py")
        return

    print("1. Завантаження покращеного датасету v2...")
    df = pd.read_csv(data_path)

    # Хронологічне сортування
    df['started_at'] = pd.to_datetime(df['started_at'])
    df = df.sort_values('started_at').reset_index(drop=True)

    df['oblast'] = df['oblast'].astype('category')

    # Оновлений список ознак (включаючи нові v2 фічі)
    feature_cols = [
        'oblast', 'hour', 'day_of_week', 'is_weekend',
        'lag_1h', 'lag_2h', 'lag_24h',
        'rolling_sum_6h', 'rolling_sum_24h',
        'rolling_delta_6h', 'country_alert_pressure'
    ]
    target_col = 'is_alert'

    X = df[feature_cols]
    y = df[target_col]

    print("2. Суворе хронологічне розділення (80% Train / 20% Test)...")
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("3. Навчання моделі LightGBM Classifier на розширеному наборі фіч...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Отримуємо ймовірності для тестової вибірки
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Застосовуємо наш перевірений поріг
    y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)

    # Розрахунок метрик
    acc = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred_custom)

    print("\n" + "=" * 50)
    print(f"ЗВІТ ЯКОСТІ МОДЕЛІ V2 (Поріг = {optimal_threshold})")
    print("=" * 50)
    print(f"Загальна точність (Accuracy):    {acc:.4f}")
    print(f"Точність прогнозу (Precision):   {precision:.4f}")
    print(f"Повнота виявлення (Recall):      {recall:.4f}")
    print(f"Збалансована міра (F1-score):    {f1:.4f}")
    print(f"Інтегрована якість (ROC-AUC):     {roc_auc:.4f}")
    print("=" * 50)

    print("\nМатриця помилок (Confusion Matrix):")
    print("                 Передбачено 0    Передбачено 1")
    print(f"Реально 0      | {cm[0][0]:<14} | {cm[0][1]}")
    print(f"Реально 1      | {cm[1][0]:<14} | {cm[1][1]}")
    print("-" * 50)

    print("\nКласифікаційний звіт:")
    print(classification_report(y_test, y_pred_custom, digits=4))

    # Важливість ознак
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Важливість ознак (Feature Importance):")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    train_and_evaluate_v2("../data/model_ready_dataset_v2.csv", optimal_threshold=0.35)