import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Гарантовано створюємо необхідні папки
Path("data").mkdir(exist_ok=True)
Path("src").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


def load_and_preprocess(file_path: str, name: str):
    path = Path(file_path)
    if not path.exists():
        print(f"[УВАГА] Файл {file_path} не знайдено.")
        return None

    print(f"Завантаження: {name}...")
    df = pd.read_csv(file_path)

    # Конвертація дат
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce')

    # Видаляємо рядки без дати початку
    df = df.dropna(subset=['started_at'])
    return df


# Шляхи до файлів
df_off = load_and_preprocess("../data/official_data_en.csv", "Офіційні дані")

if df_off is not None:
    # 1. Побудова та безпечне збереження графіка
    df_off['date'] = df_off['started_at'].dt.date
    daily_stats = df_off.groupby('date').agg(
        alert_count=('started_at', 'count')
    ).reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(pd.to_datetime(daily_stats['date']), daily_stats['alert_count'], color='#1f77b4', linewidth=1.5)
    plt.title('Динаміка кількості повітряних тривог за днями (Офіційні дані)', fontsize=14, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Кількість тривог', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    output_img = "outputs/daily_alerts_preview.png"
    plt.savefig(output_img, dpi=300)
    plt.close()  # Закриваємо фігуру, щоб уникнути помилок пам'яті
    print(f"[Успіх] Графік збережено у файл: {output_img}")

    # 2. Перетворення на погодинний часовий ряд (Hourly Resampling)
    print("\nФормування погодинного часового ряду для моделі...")

    # Встановлюємо початок тривоги як індекс для ресемплінгу
    df_off = df_off.set_index('started_at')

    # Агрегуємо у погодинні інтервали (кількість тривог за годину)
    hourly_df = df_off.resample('h').size().reset_index(name='alert_count')

    # 3. Створення календарних ознак та лагів (Lag Features)
    hourly_df['hour'] = hourly_df['started_at'].dt.hour
    hourly_df['day_of_week'] = hourly_df['started_at'].dt.dayofweek
    hourly_df['is_weekend'] = hourly_df['day_of_week'].isin([5, 6]).astype(int)

    # Додаємо лаги (значення за попередні години) та ковзні середні
    hourly_df['lag_1h'] = hourly_df['alert_count'].shift(1)
    hourly_df['lag_24h'] = hourly_df['alert_count'].shift(24)  # та сама година вчора
    hourly_df['rolling_mean_24h'] = hourly_df['alert_count'].rolling(window=24).mean()

    # Видаляємо рядки з пропуском через лаги
    hourly_df = hourly_df.dropna()

    print("Приклад підготовлених даних (features):")
    print(hourly_df.head())

    # Зберігаємо підготовлений датасет для навчання моделі
    processed_path = "../data/hourly_features.csv"
    hourly_df.to_csv(processed_path, index=False)
    print(f"\n[Успіх] Погодинні фічі збережено у {processed_path}")