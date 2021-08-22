from scipy import stats
import numpy as np

""" Известно, что рост футболистов в сборной распределен нормально с известной дисперсией 25. 
    На выборке объёма 27 выборочное среднее составило 174.2. Найдите доверительный интервал для 
    математического ожидания с надежностью 0.95.
"""

alpha = 0.05
n = 27
mean = 174.2
std = 5
t1 = stats.norm.ppf(alpha / 2)
t2 = stats.norm.ppf(1 - alpha / 2)
print("Доверительный интервал для математического ожидания:",
      (mean + t1 * std / np.sqrt(n), mean + t2 * std / np.sqrt(n)))
