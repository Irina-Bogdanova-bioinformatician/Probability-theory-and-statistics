import numpy as np
from scipy import stats

""" Продавец утверждает, что средний вес пачки печенья составляет 200 г. Из партии извлечена выборка из 10 пачек.
    Вес каждой пачки составляет: 202, 203, 199, 197, 195, 201, 200, 204, 194, 190.
    Известно, что их веса распределены нормально.
    1. Верно ли утверждение продавца, если учитывать, что уровень значимости равен 1%?
    2. Найдите P-значение для данного теста.
    
    H0: Средний вес пачки составляет 200 г.
    Н1: Средний вес пачки не составляет 200 г.
    Выбираем t-статистику.
"""

alpha = 0.01
samples = [202, 203, 199, 197, 195, 201, 200, 204, 194, 190]
n = len(samples)
mean = sum(samples) / n
t1 = stats.t.ppf(alpha / 2, df=n - 1)
t2 = stats.t.ppf(1 - alpha / 2, df=n - 1)
t = (mean - 200) / (np.std(samples, ddof=1) / np.sqrt(n))
print(f"Значение t-статистики по выборке равно {t}, критическая область: (-∞, {t1})u({t2}, +∞)")
print("Значение t-статистики по выборке не попало в критическую область - мы не отвергаем H0 и принимаеи утверждение "
      "продавца как верное")
p_left = stats.t.cdf(t, df=n - 1)
p_right = 1 - stats.t.cdf(t, df=n - 1)
p_value = 2 * min(p_left, p_right)
print(f"P-value:{p_value}, p-value > уровня значимости (0.01). Полученный ранее результат подтверждается: "
      f"мы принимаем утверждение продавца как верное", )
