import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка шрифта для отображения кириллицы
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False


# Загрузка данных
def load_cleaned_data(file_path):
    return pd.read_csv(file_path)


# Графики и анализ
def eda(df):
    # Распределение по возрасту
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], kde=True, bins=20, color='blue')
    plt.title('Распределение возраста')
    plt.xlabel('Возраст')
    plt.ylabel('Частота')
    plt.savefig('./data/age_distribution.png')
    plt.show()

    # Распределение по полу
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Gender', data=df, palette='viridis')
    plt.title('Распределение по полу')
    plt.xlabel('Пол')
    plt.ylabel('Частота')
    plt.savefig('./data/gender_distribution.png')
    plt.show()

    # Средняя сумма покупок по категориям
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y='Purchase Amount (USD)', data=df, errorbar=None, palette='pastel')
    plt.title('Средняя сумма покупок по категориям')
    plt.xlabel('Категория')
    plt.ylabel('Сумма покупок (USD)')
    plt.xticks(rotation=45)
    plt.savefig('./data/category_purchase_amount.png')
    plt.show()

    # Корреляционная матрица (исключая Customer ID)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number']).drop(columns=['Customer ID'], errors='ignore')
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица')
    plt.savefig('./data/correlation_matrix.png')
    plt.show()

    # Зависимость суммы покупок от возраста
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Age', y='Purchase Amount (USD)', data=df, hue='Gender', palette='viridis', alpha=0.7)
    plt.title('Зависимость суммы покупок от возраста')
    plt.xlabel('Возраст')
    plt.ylabel('Сумма покупок (USD)')
    plt.legend(title='Пол')
    plt.savefig('./data/age_vs_purchase.png')
    plt.show()

    # Зависимость суммы покупок от категории
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Category', y='Purchase Amount (USD)', data=df, palette='pastel')
    plt.title('Распределение суммы покупок по категориям')
    plt.xlabel('Категория')
    plt.ylabel('Сумма покупок (USD)')
    plt.xticks(rotation=45)
    plt.savefig('./data/category_vs_purchase_boxplot.png')
    plt.show()

    # Зависимость суммы покупок от пола
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Gender', y='Purchase Amount (USD)', data=df, palette='muted')
    plt.title('Распределение суммы покупок по полу')
    plt.xlabel('Пол')
    plt.ylabel('Сумма покупок (USD)')
    plt.savefig('./data/gender_vs_purchase_violinplot.png')
    plt.show()


if __name__ == "__main__":
    # Укажите путь к файлу с очищенными данными
    file_path = './data/cleaned_shopping_trends.csv'
    df = load_cleaned_data(file_path)
    eda(df)
