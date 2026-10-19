import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Чтение данных из CSV-файла
filename = input("Введите имя CSV-файла: ")
data = pd.read_csv(filename)

# Статистика по каждому столбцу
for column in data.columns:
    print(f"\nСтолбец: {column}")
    print(f"Количество: {data[column].count()}")
    print(f"Минимум: {data[column].min()}")
    print(f"Максимум: {data[column].max()}")
    print(f"Среднее: {data[column].mean():.2f}")

# Выбор столбцов
print("\nДоступные столбцы:")
for i, column in enumerate(data.columns):
    print(f"{i}: {column}")

x_col = int(input("Введите номер столбца X: "))
y_col = int(input("Введите номер столбца Y: "))

x = data.iloc[:, x_col]
y = data.iloc[:, y_col]

# Метод наименьших квадратов
x_mean = x.mean()
y_mean = y.mean()

a = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
b = y_mean - a * x_mean

print("\nПараметры регрессионной прямой:")
print(f"a = {a:.2f}")
print(f"b = {b:.2f}")
print(f"Уравнение регрессии: y = {a:.2f}x + {b:.2f}")

# Предсказанные значения
y_pred = a * x + b

# Создаём три графика
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# График 1. Исходные точки
axes[0].scatter(x, y, color="blue")
axes[0].set_title("Исходные точки")

# Данные для регрессионной прямой
x_line = np.linspace(x.min(), x.max(), 100)
y_line = a * x_line + b

# График 2. Линейная регрессия
axes[1].scatter(x, y, color="blue")
axes[1].plot(x_line, y_line, color="red")
axes[1].set_title("Линейная регрессия")

# График 3. Квадраты ошибок
axes[2].scatter(x, y, color="blue")
axes[2].plot(x_line, y_line, color="red")

for xi, yi, ypi in zip(x, y, y_pred):
    error = abs(yi - ypi)
    if error == 0:
        continue

    # Ширина квадрата
    width = error / abs(a) if a != 0 else 0
    # Точка выше линии — квадрат влево, точка ниже линии — квадрат вправо
    square_x = xi - width if yi > ypi else xi

    square = Rectangle(
        (square_x, min(yi, ypi)), width, error,
        facecolor="orange", edgecolor="orange",
        alpha=0.4, hatch="///"
    )
    axes[2].add_patch(square)
    axes[2].plot([xi, xi], [yi, ypi], color="orange", linewidth=1)

axes[2].set_title("Квадраты ошибок")

# Общие настройки графиков
for ax in axes:
    ax.set_xlabel(data.columns[x_col])
    ax.set_ylabel(data.columns[y_col])
    ax.grid(True)

plt.tight_layout()
plt.show()