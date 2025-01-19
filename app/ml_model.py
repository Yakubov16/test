import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Настройка отображения графиков
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

# Функция предварительной обработки данных
def preprocess_data(df):
    """
    Предобрабатывает данные: кодирует категориальные переменные и стандартизирует числовые признаки.

    Args:
        df: Исходный датафрейм.

    Returns:
        df: Обработанный датафрейм.
    """
    df = df.copy()

    # Кодирование категориальных переменных
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder

    # Нормализация числовых данных
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
    """
    Оценивает модель по метрикам MAE, RMSE и R^2.

    Args:
        model: Обученная модель.
        X_test: Признаки тестовой выборки.
        y_test: Истинные значения тестовой выборки.

    Returns:
        predictions: Прогнозы модели.
        metrics: Словарь с метриками качества.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R^2': r2
    }
    return predictions, metrics

# Функция для создания графиков прогнозирования
def create_forecast_plots(X_test, y_test, predictions, model, model_name, output_dir):
    """
    Создает графики для анализа качества прогнозов и сохраняет их в указанной папке.

    Args:
        X_test: Признаки тестовой выборки.
        y_test: Истинные значения тестовой выборки.
        predictions: Прогнозы модели.
        model: Обученная модель.
        model_name: Название модели.
        output_dir: Папка для сохранения графиков.
    """
    # Проверяем существование папки для графиков
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Scatterplot: Истинные значения vs Прогнозы
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Истинные значения vs Прогнозы ({model_name})')
    plt.xlabel('Истинные значения')
    plt.ylabel('Прогнозы')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{model_name}_scatter.png'))
    plt.close()

    # Остатки: Истинные значения - Прогнозы
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=20, color='green')
    plt.title(f'Распределение остатков ({model_name})')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{model_name}_residuals.png'))
    plt.close()

    # Feature Importance (если применимо)
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importance = pd.Series(model.feature_importances_, index=X_test.columns)
        importance.nlargest(5).plot(kind='barh', color='skyblue')  # Показываем только 5 самых значимых факторов
        plt.title(f'Важность признаков ({model_name})')
        plt.xlabel('Значимость')
        plt.ylabel('Признак')
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importance.png'))
        plt.close()

    # Lineplot: Прогнозы vs Истинные значения
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label='Истинные значения', color='blue')
    plt.plot(range(len(predictions)), predictions, label='Прогнозы', color='orange')
    plt.title(f'Сравнение прогнозов и истинных значений ({model_name})')
    plt.xlabel('Наблюдения')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{model_name}_lineplot.png'))
    plt.close()

# Основная функция

def train_and_evaluate(df):
    """
    Обучает модели машинного обучения и оценивает их качество.

    Args:
        df: Датафрейм с данными.

    Returns:
        results: Словарь с метриками качества для каждой модели.
    """
    # Предварительная обработка данных
    df = preprocess_data(df)

    # Целевая переменная и признаки
    X = df.drop(columns=['Purchase Amount (USD)', 'Customer ID'])
    y = df['Purchase Amount (USD)']

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Модели для обучения
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    # Обучение и оценка моделей
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions, metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        # Создание графиков для каждой модели
        create_forecast_plots(X_test, y_test, predictions, model, name, output_dir='./forecast')

    return results

if __name__ == "__main__":
    # Загрузка данных
    file_path = './data/cleaned_shopping_trends.csv'
    df = pd.read_csv(file_path)

    # Обучение и оценка моделей
    results = train_and_evaluate(df)

    # Вывод результатов
    for model_name, metrics in results.items():
        print(f"\nМодель: {model_name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
