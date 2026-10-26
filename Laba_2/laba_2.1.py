import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# Загрузка данных
filename = input("Введите имя CSV-файла (например, lab2_1_DS.csv): ")
data = pd.read_csv(filename)

# Выбор столбцов
print("\nДоступные столбцы:")
for i, column in enumerate(data.columns):
    print(f"{i}: {column}")

x_col = int(input("Введите номер столбца X: "))
y_col = int(input("Введите номер столбца Y: "))

x_name = data.columns[x_col]
y_name = data.columns[y_col]

X = data[[x_name]]
y = data[y_name]

# Разделение данных на обучающую и тестовую выборки (берем 30% на тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание моделей
models = {
    "Простая линейная регрессия": LinearRegression(),
    "Ridge": Ridge(),
    "LASSO": Lasso(),
    "Elastic Net": ElasticNet()
}

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label='Обучающие данные')
plt.scatter(X_test, y_test, color='gray', marker='x', label='Тестовые данные')

x_line = np.linspace(X.min().values[0], X.max().values[0], 100)
x_line_df = pd.DataFrame(x_line, columns=[x_name])

# Разные настройки для линий, чтобы они не сливались
colors = ['blue', 'red', 'green', 'orange']
linestyles = ['-', '--', '-.', ':']
linewidths = [5, 4, 3, 2]

metrics_results = []

print("\n--- Уравнения моделей ---")
for (name, model), color, ls, lw in zip(models.items(), colors, linestyles, linewidths):
    # Обучаем модель только на обучающей выборке
    model.fit(X_train, y_train)

    # Выводим параметры модели в виде уравнения y = ax + b
    print(f"{name} | Уравнение: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    # Делаем предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    # Расчет метрик
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Сохраняем результаты для таблицы
    metrics_results.append({
        "Модель": name,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    })

    # Предсказание для отрисовки линии на графике
    y_line = model.predict(x_line_df)
    plt.plot(x_line, y_line, color=color, linestyle=ls, linewidth=lw, label=name)

print("\n--- Таблица метрик ---")
metrics_df = pd.DataFrame(metrics_results)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# График
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.title('Сравнение моделей регрессии (с разделением данных)')
plt.legend()
plt.grid(True)
plt.show()