import pandas as pd
from pathlib import Path


def build_ml_dataset(file_path: str, output_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл {file_path} не знайдено.")
        return

    print("1. Завантаження та очищення сирих даних...")
    df = pd.read_csv(file_path)

    # Залишаємо лише необхідні колонки та конвертуємо дати у UTC
    df = df[['oblast', 'started_at', 'finished_at']].dropna(subset=['oblast', 'started_at'])
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce')
    df = df.dropna(subset=['started_at']).sort_values('started_at')

    print("2. Генерація повного погодинного сіткового календаря для кожного регіону...")
    # Визначаємо глобальний діапазон часу від мінімальної до максимальної дати в даних
    min_time = df['started_at'].min().floor('h')
    max_time = df['started_at'].max().ceil('h')

    all_hours = pd.date_range(start=min_time, end=max_time, freq='h')
    oblasts = df['oblast'].unique()

    # Створюємо базовий каркас: усі комбінації (регіон x година)
    index_grid = pd.MultiIndex.from_product([oblasts, all_hours], names=['oblast', 'started_at'])
    grid_df = pd.DataFrame(index=index_grid).reset_index()

    print("3. Розрахунок цільової змінної (is_alert)...")
    # Визначаємо, в які години була хоча б одна активна тривога для кожної області
    # Оптимальний спосіб через маркування інтервалів
    df['hour_start'] = df['started_at'].dt.floor('h')

    # Створюємо масив годин, у які реально була тривога для конкретної області
    # (для спрощення та швидкості на вікенд-проєкті розмітимо години початку або перетину)
    active_alerts = df[['oblast', 'hour_start']].drop_duplicates()
    active_alerts['is_alert'] = 1

    # Об'єднуємо каркас із реатіями тривог
    merged_df = pd.merge(grid_df, active_alerts, left_on=['oblast', 'started_at'], right_on=['oblast', 'hour_start'],
                         how='left')
    merged_df['is_alert'] = merged_df['is_alert'].fillna(0).astype(int)
    merged_df = merged_df.drop(columns=['hour_start'])

    # Сортуємо суворо за регіоном і часом для безпечного розрахунку лагів
    merged_df = merged_df.sort_values(by=['oblast', 'started_at']).reset_index(drop=True)

    print("4. Інжиніринг ознак (Календар, Лаги, Ковзні вікна)...")
    # Календарні фічі
    merged_df['hour'] = merged_df['started_at'].dt.hour
    merged_df['day_of_week'] = merged_df['started_at'].dt.dayofweek
    merged_df['is_weekend'] = merged_df['day_of_week'].isin([5, 6]).astype(int)

    # Групуємо по кожній області окремо, щоб лаги не "перетікали" з одного регіону в інший
    group = merged_df.groupby('oblast', group_keys=False)

    # Лагові ознаки (статус тривоги за попередні години)
    merged_df['lag_1h'] = group['is_alert'].shift(1)
    merged_df['lag_2h'] = group['is_alert'].shift(2)
    merged_df['lag_24h'] = group['is_alert'].shift(24)  # та сама година вчора

    # Ковзні суми за останні 6 та 24 години (скільки годин лихоманило область)
    merged_df['rolling_sum_6h'] = group['is_alert'].rolling(window=6, min_periods=1).sum().reset_index(0, drop=True)
    merged_df['rolling_sum_24h'] = group['is_alert'].rolling(window=24, min_periods=1).sum().reset_index(0, drop=True)

    # Видаляємо перші рядки кожної області, де лаги містять пропуски (NaN)
    merged_df = merged_df.dropna().reset_index(drop=True)

    # Зберігаємо готовий датасет
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"\n[Успіх] Фінальний датасет для моделювання збережено у: {output_path}")
    print(f"Загальна кількість рядків: {merged_df.shape[0]}, стовпчиків: {merged_df.shape[1]}")
    print(merged_df.head(10))


if __name__ == "__main__":
    build_ml_dataset("../data/official_data_en.csv", "../data/model_ready_dataset.csv")