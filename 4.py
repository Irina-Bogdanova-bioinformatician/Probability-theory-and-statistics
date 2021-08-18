import numpy as np
from scipy import stats

""" Дана выборка диаметров подшипников из примера 1 с занятия 5:
    samples = [0.6603, 0.9466, 0.5968, 1.3792, 1.5481, 0.7515, 1.0681, 1.1134, 1.2088, 1.701 , 1.0282, 
    1.3579, 1.0191, 1.1784, 1.1168, 1.1372, 0.7273, 1.3958, 0.8665, 1.5112, 1.161 , 1.0232, 1.0865, 1.02 ]

    Предполагая, что диаметры подшипников распределены нормально, проверьте гипотезу о том, что дисперсия 
    случайной величины равна 0.0625 при уровне значимости alpha = 0.05. 
    
    H0: Дисперсия случайной величины равна 0.0625.
    Н1: Дисперсия случайной величины не равна 0.0625.
    
    Что для этого нужно знать:
    1. Альтернативная гипотеза двухсторонняя.
    2. Статистика для теста: H = (n - 1) * sample_variance / variance, где n - число элементов в выборке, 
    sample_variance - несмещённая оценка дисперсии, variance - утверждаемая нулевой гипотезой дисперсия.
    3. Эта статистика в предположении верности нулевой гипотезы имеет распределение хи-квадрат 
    с параметром df = n - 1. Её квантили можно найти с помощью функции scipy.stats.chi2.ppf
"""

samples = [0.6603, 0.9466, 0.5968, 1.3792, 1.5481, 0.7515, 1.0681, 1.1134, 1.2088, 1.701, 1.0282, 1.3579,
           1.0191, 1.1784, 1.1168, 1.1372, 0.7273, 1.3958, 0.8665, 1.5112, 1.161, 1.0232, 1.0865, 1.02]
alpha = 0.05
n = len(samples)
t1 = stats.chi2.ppf(alpha / 2, df=n - 1)
t2 = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
h = (n - 1) * np.var(samples, ddof=1) / 0.0625
print(f"Значение H-статистики по выборке равно {h}, критическая область: (-∞, {t1})u({t2}, +∞)")
print("Значение H-статистики по выборке не попало в критическую область - мы принимаем нулевую гипотезу")
p_left = stats.chi2.cdf(h, df=n - 1)
p_right = 1 - stats.chi2.cdf(h, df=n - 1)
p_value = 2 * min(p_left, p_right)
print(f"P-value:{p_value}, p-value > уровня значимости (0.05). Полученный ранее результат подтверждается: "
      f"мы принимаем нулевую гипотезу", )
