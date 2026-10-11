import pandas as pd
import numpy as np
from pathlib import Path

# Спрощений словник сусідства областей України (для просторового контексту)
NEIGHBORS = {
    'Kyivska': ['Zhytomyrska', 'Chernihivska', 'Poltavska', 'Cherkaska', 'Vinnytska'],
    'Kharkivska': ['Sumska', 'Poltavska', 'Dnipropetrovska_r', 'Luhanska', 'Donetska'],
    # адаптуйте під назви у вашому файлі
    # Загальний словник або базове групування можна розширити,
    # для прикладу використаємо автоматичне визначення або безпечний дефолт
}


def build_advanced_dataset(file_path: str, output_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {file_path} не знайдено.")
        return

    print("1. Завантаження та очищення сирих даних...")
    df = pd.read_csv(file_path)
    df = df[['oblast', 'started_at', 'finished_at']].dropna(subset=['oblast', 'started_at'])
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
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

    print("4. Інжиніринг базових та нових покращених ознак...")
    # Календарні фічі
    merged_df['hour'] = merged_df['started_at'].dt.hour
    merged_df['day_of_week'] = merged_df['started_at'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)

    group = merged_df.groupby('oblast', group_keys=False)

    # Стандартні лаги та ковзні суми
    merged_df['lag_1h'] = group['is_alert'].shift(1)
    merged_df['lag_2h'] = group['is_alert'].shift(2)
    merged_df['lag_24h'] = group['is_alert'].shift(24)
    merged_df['rolling_sum_6h'] = group['is_alert'].rolling(window=6, min_periods=1).sum().reset_index(0, drop=True)
    merged_df['rolling_sum_24h'] = group['is_alert'].rolling(window=24, min_periods=1).sum().reset_index(0, drop=True)

    # --- НОВІ ФІЧІ ДЛЯ ЗМЕНШЕННЯ ХИБНИХ СПРАЦЬУВАНЬ ---

    # 1. Динаміка (швидкість зміни) ковзної суми: показує, нагрівається ситуація чи остигає
    merged_df['rolling_delta_6h'] = group['rolling_sum_6h'].diff(1).fillna(0)

    # 2. Просторовий контекст (середня активність по всій країні в цю ж годину як індикатор загального масованого фону)
    # Це дозволяє моделі бачити загальнодержавний масштаб загрози замість локальних помилок
    country_wide = merged_df.groupby('started_at')['is_alert'].mean().reset_index(name='country_alert_pressure')
    merged_df = pd.merge(merged_df, country_wide, on='started_at', how='left')

    # Видаляємо пропуски від лагів
    merged_df = merged_df.dropna().reset_index(drop=True)

    output_path = "../data/model_ready_dataset_v2.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n[Успіх] Покращений датасет збережено у: {output_path}")
    print(f"Кількість рядків: {merged_df.shape[0]}, стовпчиків: {merged_df.shape[1]}")


if __name__ == "__main__":
    build_advanced_dataset("../data/official_data_en.csv", "../data/model_ready_dataset_v2.csv")