import numpy as np
import matplotlib.pyplot as plt

salaries = [100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75, 65, 84, 90, 150]
quartiles = np.quantile(salaries, [0.25, 0.75])
print(f"Первый квартиль равен {quartiles[0]}, третий квартиль равен {quartiles[1]}")
iqr = quartiles[1] - quartiles[0]
print("Интерквартильное расстояние равно", iqr)
boxplot_range = [quartiles[0] - iqr, quartiles[1] + iqr]
print("Диапазон значений, лежащих в пределах boxplot (включая усы):", boxplot_range)
outliers = [x for x in salaries if x < boxplot_range[0] or x > boxplot_range[1]]
print("Список выбросов:", outliers)
print("Доля выбросов равна", len(outliers) / len(salaries))
plt.boxplot(salaries)
plt.show()
