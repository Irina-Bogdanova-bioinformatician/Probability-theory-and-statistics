import pandas as pd
import numpy as np

""" Для расчета среднего арифметического, среднего квадратичного отклонения,
    смещенной и несмещенной оценки дисперсий для данной выборки использовали библиотеки pandas и numpy.
"""

salaries = [100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75, 65, 84, 90, 150]
df = pd.DataFrame(salaries)
mean_salary = df.sum() / df.count()
print("Среднее арифметическое:", mean_salary)
print("Среднее арифметическое, полученное с помощью метода .mean:", df.mean())
biased_variance = ((df - mean_salary) ** 2).sum() / df.count()
print("Смещённая оценка выборочной дисперсии:", biased_variance)
print("Смещённая оценка выборочной дисперсии, полученная с помощью метода .var:", df.var(ddof=0))
unbiased_variance = ((df - mean_salary) ** 2).sum() / (df.count() - 1)
print("Несмещённая оценка выборочной дисперсии:", unbiased_variance)
print("Несмещённая оценка выборочной дисперсии, полученная с помощью метода .var:", df.var(ddof=1))
standard_dev_biased = np.sqrt(biased_variance)
print("Смещённое среднее квадратичное отклонение:", standard_dev_biased)
print("Смещённое среднее квадратичное отклонение, полученное с помощью метода .std:", df.std(ddof=0))
standard_dev_unbiased = np.sqrt(unbiased_variance)
print("Несмещённое среднее квадратичное отклонение:", standard_dev_unbiased)
print("Несмещённое среднее квадратичное отклонение, полученное с помощью метода .std:", df.std(ddof=1))
