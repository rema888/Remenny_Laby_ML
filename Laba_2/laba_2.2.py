import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Загрузка данных
filename = input("Введите имя CSV-файла (например, lab2_2_DS.csv): ")
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

# Создание моделей (используем разные степени полинома)
degrees = [2, 3, 4, 5]

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label='Обучающие данные')
plt.scatter(X_test, y_test, color='gray', marker='x', label='Тестовые данные')

x_line = np.linspace(X.min().values[0], X.max().values[0], 100)
x_line_df = pd.DataFrame(x_line, columns=[x_name])

# Разные настройки для линий, чтобы они не сливались
colors = ['blue', 'green', 'red', 'orange']
linestyles = ['-', '--', '-.', ':']
linewidths = [3, 2, 2, 2]

metrics_results = []

for degree, color, ls, lw in zip(degrees, colors, linestyles, linewidths):
    # Преобразование признаков в полиномиальные
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # Обучаем модель только на обучающей выборке
    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    # Делаем предсказания на тестовой выборке
    y_pred = model.predict(X_poly_test)

    # Расчет метрик
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Сохраняем результаты для таблицы
    metrics_results.append({
        "Степень полинома": degree,
        "MSE": mse,
        "R2": r2,
        "MAPE": mape
    })

    # Предсказание для отрисовки линии на графике
    X_line_poly = poly.transform(x_line_df)
    y_line = model.predict(X_line_poly)
    plt.plot(x_line, y_line, color=color, linestyle=ls, linewidth=lw, label=f'Степень {degree}')

print("\n--- Таблица метрик ---")
metrics_df = pd.DataFrame(metrics_results)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# График
plt.xlabel(x_name)
plt.ylabel(y_name)
plt.title('Сравнение моделей полиномиальной регрессии')
plt.legend()
plt.grid(True)
plt.show()