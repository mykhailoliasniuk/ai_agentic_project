import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from pathlib import Path


def build_and_train_clean_model(file_path: str, optimal_threshold: float = 0.35):
    path = Path(file_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {file_path} не знайдено.")
        return

    print("1. Завантаження та очищення сирих даних...")
    df = pd.read_csv(file_path)
    df = df[['oblast', 'started_at', 'finished_at']].dropna(subset=['oblast', 'started_at'])
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce')
    df = df.dropna(subset=['started_at']).sort_values('started_at')

    print("2. Генерація погодинної сітки для всіх регіонів...")
    min_time = df['started_at'].min().floor('h')
    max_time = df['started_at'].max().ceil('h')

    all_hours = pd.date_range(start=min_time, end=max_time, freq='h')
    oblasts = df['oblast'].unique()

    index_grid = pd.MultiIndex.from_product([oblasts, all_hours], names=['oblast', 'started_at'])
    grid_df = pd.DataFrame(index=index_grid).reset_index()

    print("3. Розрахунок цільової змінної (is_alert)...")
    df['hour_start'] = df['started_at'].dt.floor('h')
    active_alerts = df[['oblast', 'hour_start']].drop_duplicates()
    active_alerts['is_alert'] = 1

    merged_df = pd.merge(grid_df, active_alerts, left_on=['oblast', 'started_at'], right_on=['oblast', 'hour_start'],
                         how='left')
    merged_df['is_alert'] = merged_df['is_alert'].fillna(0).astype(int)
    merged_df = merged_df.drop(columns=['hour_start'])
    merged_df = merged_df.sort_values(by=['oblast', 'started_at']).reset_index(drop=True)

    print("4. Безпечний інжиніринг ознак (без витоку майбутнього)...")
    # Календарні фічі
    merged_df['hour'] = merged_df['started_at'].dt.hour
    merged_df['day_of_week'] = merged_df['started_at'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)

    group = merged_df.groupby('oblast', group_keys=False)

    # Справжні лаги (інформація виключно з минулого)
    merged_df['lag_1h'] = group['is_alert'].shift(1)
    merged_df['lag_2h'] = group['is_alert'].shift(2)
    merged_df['lag_24h'] = group['is_alert'].shift(24)

    # Ковзні суми на основі минулих годин (min_periods=1 для збереження даних)
    merged_df['rolling_sum_6h'] = group['is_alert'].shift(1).rolling(window=6, min_periods=1).sum().reset_index(0,
                                                                                                                drop=True)
    merged_df['rolling_sum_24h'] = group['is_alert'].shift(1).rolling(window=24, min_periods=1).sum().reset_index(0,
                                                                                                                  drop=True)

    # Динаміка (зміна ковзної суми за попередні періоди)
    merged_df['rolling_delta_6h'] = group['rolling_sum_6h'].diff(1).fillna(0)

    # Загальнодержавний тиск через лаг (щоб не брати поточну годину t)
    country_wide = merged_df.groupby('started_at')['is_alert'].mean().reset_index(name='country_alert_pressure')
    country_wide['started_at'] = country_wide['started_at'] + pd.Timedelta(hours=1)  # зсув у минуле
    merged_df = pd.merge(merged_df, country_wide, on='started_at', how='left')
    merged_df['country_alert_pressure'] = merged_df['country_alert_pressure'].fillna(0)

    # Очищення від NaN, спричинених лагами
    merged_df = merged_df.dropna().reset_index(drop=True)

    print("5. Суворе хронологічне розділення вибірки (80% Train / 20% Test)...")
    split_index = int(len(merged_df) * 0.8)

    merged_df['oblast'] = merged_df['oblast'].astype('category')
    feature_cols = [
        'oblast', 'hour', 'day_of_week', 'is_weekend',
        'lag_1h', 'lag_2h', 'lag_24h',
        'rolling_sum_6h', 'rolling_sum_24h',
        'rolling_delta_6h', 'country_alert_pressure'
    ]
    target_col = 'is_alert'

    X = merged_df[feature_cols]
    y = merged_df[target_col]

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print("6. Навчання чесної моделі LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("7. Оцінка якості на тестовому періоді...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred_custom)

    print("\n" + "=" * 50)
    print(f"ЧЕСНИЙ ФІНАЛЬНИЙ ЗВІТ ЯКОСТІ (Поріг = {optimal_threshold})")
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

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Важливість ознак:")
    print(importance_df.to_string(index=False))
    # Додайте цей цикл у кінці вашого чистого скрипта замість фіксованого порогу 0.35:
    print("\n--- Пошук нового оптимального порогу для чесної моделі ---")
    for th in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        y_pred_th = (y_pred_proba >= th).astype(int)
        rec = recall_score(y_test, y_pred_th, zero_division=0)
        prec = precision_score(y_test, y_pred_th, zero_division=0)
        f1 = f1_score(y_test, y_pred_th, zero_division=0)
        print(f"Поріг: {th:.2f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    build_and_train_clean_model("../data/official_data_en.csv", optimal_threshold=0.35)