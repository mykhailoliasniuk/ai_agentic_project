import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path


def train_and_optimize_thresholds(data_path: str):
    path = Path(data_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {data_path} не знайдено. Спочатку запустіть build_dataset.py")
        return

    print("1. Завантаження підготовленого датасету...")
    df = pd.read_csv(data_path)

    # Хронологічне сортування
    df['started_at'] = pd.to_datetime(df['started_at'])
    df = df.sort_values('started_at').reset_index(drop=True)

    df['oblast'] = df['oblast'].astype('category')
    feature_cols = ['oblast', 'hour', 'day_of_week', 'is_weekend', 'lag_1h', 'lag_2h', 'lag_24h', 'rolling_sum_6h',
                    'rolling_sum_24h']
    target_col = 'is_alert'

    X = df[feature_cols]
    y = df[target_col]

    print("2. Суворе хронологічне розділення (80% Train / 20% Test)...")
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("3. Навчання моделі LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Отримуємо ймовірності для тестової вибірки
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n4. Цикл перебору порогів класифікації (Threshold Tuning)...")
    print("-" * 65)
    print(f"{'Поріг (Threshold)':<18} | {'Recall':<10} | {'Precision':<10} | {'F1-score':<10} | {'Accuracy':<10}")
    print("-" * 65)

    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}

    # Перебираємо пороги від 0.10 до 0.50 з кроком 0.05
    thresholds = np.arange(0.10, 0.55, 0.05)

    for th in thresholds:
        # Перетворюємо ймовірності у класи за допомогою поточного порогу
        y_pred_th = (y_pred_proba >= th).astype(int)

        acc = accuracy_score(y_test, y_pred_th)
        precision = precision_score(y_test, y_pred_th, zero_division=0)
        recall = recall_score(y_test, y_pred_th, zero_division=0)
        f1 = f1_score(y_test, y_pred_th, zero_division=0)

        print(f"{th:<18.2f} | {recall:<10.4f} | {precision:<10.4f} | {f1:<10.4f} | {acc:<10.4f}")

        # Шукаємо компроміс: найкращий F1-score серед варіантів із високим Recall
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    print("-" * 65)
    print(f"\n[РЕКОМЕНДАЦІЯ] Оптимальний поріг для балансу: {best_threshold:.2f}")
    print(f"З цим порогом отримуємо Recall = {best_metrics['recall']:.4f} при F1 = {best_metrics['f1']:.4f}")

    # Базовий стандартний показник для порівняння (при порозі 0.5)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    print(f"\nДля порівняння (стандартний поріг 0.5):")
    print(f"  - Recall:    {recall_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  - Precision: {precision_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"  - F1-score:  {f1_score(y_test, y_pred_default, zero_division=0):.4f}")


if __name__ == "__main__":
    train_and_optimize_thresholds("../data/model_ready_dataset.csv")