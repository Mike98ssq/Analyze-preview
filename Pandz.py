import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.tree import *

# Генерация случайных данных
num_products = 10
num_days = 30
products = [f"Product {i+1}" for i in range(num_products)]
days = list(range(1, num_days + 1))

data = {
    'Product': np.random.choice(products, num_products * num_days),
    'Day': np.random.choice(days, num_products * num_days),
    'Sales': np.random.randint(10, 100, num_products * num_days)
}

df = pd.DataFrame(data)

# Анализ данных
def analyze_sales(df):
    print("Таблица данных:")
    print(df.head())

    sales_aggregations = {
        'Общие продажи по продуктам': df.groupby('Product')['Sales'].sum(),
        'Средние продажи по дням': df.groupby('Day')['Sales'].mean(),
        'Максимальные продажи по продуктам': df.groupby('Product')['Sales'].max(),
        'Минимальные продажи по продуктам': df.groupby('Product')['Sales'].min()
    }

    for title, result in sales_aggregations.items():
        print(f"\n{title}:\n{result}")

    correlation = df['Day'].corr(df['Sales'])
    print(f"\nКорреляция между днем и продажами: {correlation}")

    return sales_aggregations['Общие продажи по продуктам'], sales_aggregations['Средние продажи по дням']

total_sales_by_product, average_sales_by_day = analyze_sales(df)

# Визуализация данных
def visualize_data(total_sales_by_product, average_sales_by_day, df):  
    #типы графиков 
    # types_plot = ['area',
    #               'bar',
    #               'barh',
    #               'box',
    #               'density',
    #               'hexbin',
    #               'hexbin',
    #               'hist',
    #               'kde',
    #               'line',
    #               'pie',
    #               'scatter']
    # Графики
    plt.subplot(2, 2, 1)
    total_sales_by_product.plot(kind='bar')
    plt.title('Общие продажи по продуктам')
    plt.xlabel('Продукты')
    plt.ylabel('Суммарные продажи')

    plt.subplot(2, 2, 2)
    average_sales_by_day.plot(kind='line')
    plt.title('Средние продажи по дням')
    plt.xlabel('День')
    plt.ylabel('Средние продажи')

    plt.subplot(2, 2, 3)
    df['Sales'].hist(bins=30)
    plt.title('Распределение продаж')
    plt.xlabel('Продажи')
    plt.ylabel('Частота')

    plt.subplot(2, 2, 4)
    plt.scatter(df['Day'], df['Sales'], alpha=0.4)

    # Линейная регрессия
    model = LinearRegression().fit(df[['Day']], df['Sales'])
    plt.plot(df['Day'], model.predict(df[['Day']]), color='red', linewidth=2, label='Линия регрессии')
    plt.title('День vs Продажи с линией регрессии')
    plt.xlabel('День')
    plt.ylabel('Продажи')
    plt.legend()

    plt.tight_layout()
    plt.show()

visualize_data(total_sales_by_product, average_sales_by_day, df)

# Тепловая карта
def plot_heatmap(df):
    pivot_table = df.pivot_table(values='Sales', index='Product', columns='Day', aggfunc='min')
    sns.heatmap(pivot_table, annot = True)
    plt.title('Тепловая карта продаж по продуктам и дням')
    plt.xlabel('День')
    plt.ylabel('Продукт')
    plt.show()

plot_heatmap(df)

# Предсказание продаж
def predict_sales(df):
    df_grouped = df.groupby(['Day', 'Product']).sum().reset_index()
    X_train, X_test, y_train, y_test = train_test_split(df_grouped[['Day']], df_grouped['Sales'], test_size=0.2)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nСреднеквадратичная ошибка: {mean_squared_error(y_test, y_pred):.2f}",
          f"\nКоэффициент детерминации (R^2): {r2_score(y_test, y_pred):.2f}")


predict_sales(df)