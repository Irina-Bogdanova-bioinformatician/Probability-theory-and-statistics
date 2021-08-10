import numpy as np

""" Для расчета среднего арифметического, среднего квадратичного отклонения,
    смещенной и несмещенной оценки дисперсий для данной выборки использовали библиотеку numpy.
"""

salaries = [100, 80, 75, 77, 89, 33, 45, 25, 65, 17, 30, 24, 57, 55, 70, 75, 65, 84, 90, 150]
mean_salary = np.sum(salaries) / len(salaries)
print("Среднее арифметическое:", mean_salary)
print("Среднее арифметическое, полученное с помощью метода .mean:", np.mean(salaries))
biased_variance = np.sum([(x - mean_salary) ** 2 for x in salaries]) / len(salaries)
print("Смещённая оценка выборочной дисперсии:", biased_variance)
print("Смещённая оценка выборочной дисперсии, полученная с помощью метода .var:", np.var(salaries))
unbiased_variance = np.sum([(x - mean_salary) ** 2 for x in salaries]) / (len(salaries) - 1)
print("Несмещённая оценка выборочной дисперсии:", unbiased_variance)
print("Несмещённая оценка выборочной дисперсии, полученная с помощью метода .var:", np.var(salaries, ddof=1))
standard_dev_biased = np.sqrt(biased_variance)
print("Смещённое среднее квадратичное отклонение:", standard_dev_biased)
print("Смещённое среднее квадратичное отклонение, полученное с помощью метода .std:", np.std(salaries))
standard_dev_unbiased = np.sqrt(unbiased_variance)
print("Несмещённое среднее квадратичное отклонение:", standard_dev_unbiased)
print("Несмещённое среднее квадратичное отклонение, полученное с помощью метода .std:", np.std(salaries, ddof=1))
