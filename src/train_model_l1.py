import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from pathlib import Path


def evaluate_final_model(data_path: str, optimal_threshold: float = 0.35):
    path = Path(data_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {data_path} не знайдено.")
        return

    print("1. Завантаження датасету та підготовка вибірки...")
    df = pd.read_csv(data_path)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df = df.sort_values('started_at').reset_index(drop=True)

    df['oblast'] = df['oblast'].astype('category')
    feature_cols = ['oblast', 'hour', 'day_of_week', 'is_weekend', 'lag_1h', 'lag_2h', 'lag_24h', 'rolling_sum_6h',
                    'rolling_sum_24h']
    target_col = 'is_alert'

    X = df[feature_cols]
    y = df[target_col]

    # Суворе хронологічне розділення 80/20
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("2. Навчання фінальної моделі LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Отримуємо ймовірності для тесту
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Застосовуємо обраний оптимальний поріг
    y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)

    # Розрахунок метрик
    acc = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred_custom)

    print("\n" + "=" * 50)
    print(f"ФІНАЛЬНИЙ ЗВІТ ЯКОСТІ МОДЕЛІ (Поріг класифікації = {optimal_threshold})")
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

    print("\nДетальний класифікаційний звіт (Classification Report):")
    print(classification_report(y_test, y_pred_custom, digits=4))

    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Важливість ознак:")
    print(importance_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_final_model("../data/model_ready_dataset.csv", optimal_threshold=0.35)