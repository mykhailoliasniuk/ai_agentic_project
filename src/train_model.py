import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path


def train_and_evaluate_model(data_path: str):
    path = Path(data_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {data_path} не знайдено. Спочатку запустіть build_dataset.py")
        return

    print("1. Завантаження підготовленого датасету...")
    df = pd.read_csv(data_path)

    # Гарантуємо суворе хронологічне сортування за часом
    df['started_at'] = pd.to_datetime(df['started_at'])
    df = df.sort_values('started_at').reset_index(drop=True)

    print(f"Загальна кількість рядків: {df.shape[0]}")

    # Визначаємо ознаки (features) та цільову змінну (target)
    # Зберігаємо категоріальний тип для 'oblast', щоб LightGBM коректно з ним працював
    df['oblast'] = df['oblast'].astype('category')

    feature_cols = ['oblast', 'hour', 'day_of_week', 'is_weekend', 'lag_1h', 'lag_2h', 'lag_24h', 'rolling_sum_6h',
                    'rolling_sum_24h']
    target_col = 'is_alert'

    X = df[feature_cols]
    y = df[target_col]

    print("2. Суворе хронологічне розділення (80% минуле для Train, 20% майбутнє для Test)...")
    split_index = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    train_start_date = df['started_at'].iloc[0]
    train_end_date = df['started_at'].iloc[split_index - 1]
    test_start_date = df['started_at'].iloc[split_index]
    test_end_date = df['started_at'].iloc[-1]

    print(f"   - Тренувальний період: від {train_start_date} до {train_end_date} ({len(X_train)} рядків)")
    print(f"   - Тестовий період (майбутнє): від {test_start_date} до {test_end_date} ({len(X_test)} рядків)")

    print("\n3. Навчання моделі LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    # Навчаємо виключно на тренувальній вибірці
    model.fit(X_train, y_train)

    print("\n4. Оцінка якості на тестовій вибірці (найсвіжіші дані)...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Розрахунок метрик
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТИ ОЦІНКИ МОДЕЛІ НА ТЕСТІ:")
    print("=" * 40)
    print(f"Accuracy (Точність загальна):   {acc:.4f}")
    print(f"Precision (Точність прогнозу 1): {precision:.4f}")
    print(f"Recall (Повнота виявлення 1):    {recall:.4f}")
    print(f"F1-score (Збалансована метрика):  {f1:.4f}")
    print(f"ROC-AUC:                         {roc_auc:.4f}")
    print("=" * 40)

    # Виводимо важливість ознак (Feature Importance), щоб побачити, що найбільше впливало на рішення моделі
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nВажливість ознак для моделі:")
    print(importance_df)


if __name__ == "__main__":
    train_and_evaluate_model("../data/model_ready_dataset.csv")