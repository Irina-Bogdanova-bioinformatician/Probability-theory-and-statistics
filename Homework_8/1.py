import numpy as np
from scipy import stats

""" Провести дисперсионный анализ для определения того, есть ли различия среднего роста среди взрослых 
    футболистов, хоккеистов и штангистов. Даны значения роста в трех группах случайно выбранных спортсменов:

    football_players = [173, 175, 180, 178, 177, 185, 183, 182]
    hockey_players = [177, 179, 180, 188, 177, 172, 171, 184, 180]
    lifters = [172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170]
    
    H0: различий среднего роста среди взрослых футболистов, хоккеистов и штангистов нет
    H1: различия есть
    alpha = 0.05
"""

football = np.array([173, 175, 180, 178, 177, 185, 183, 182])
hockey = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
lifters = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])
n_football = football.shape[0]
n_hockey = hockey.shape[0]
n_lifters = lifters.shape[0]
football_mean = football.mean()
hockey_mean = hockey.mean()
lifters_mean = lifters.mean()
print(f"Средний рост футболстов: {football_mean}, хоккеистов: {hockey_mean}, штангистов: {lifters_mean}")
sportsmen = np.concatenate([football, hockey, lifters])
sportsmen_mean = sportsmen.mean()
ss_b = n_football * (football_mean - sportsmen_mean) ** 2 + n_hockey * (hockey_mean - sportsmen_mean) ** 2 + \
       n_lifters * (lifters_mean - sportsmen_mean) ** 2
ss_w = ((football - football_mean) ** 2).sum() + ((hockey - hockey_mean) ** 2).sum() + \
       ((lifters - lifters_mean) ** 2).sum()
k = 3
n = n_lifters + n_hockey + n_football
k1 = k - 1
k2 = n - k
sigma2_b = ss_b / k1
sigma2_w = ss_w / k2
f = sigma2_b / sigma2_w
alpha = 0.05
f_cr = stats.f.ppf(1 - alpha, k1, k2)
print(f"Критическая область: ({f_cr}, +∞)")
print(f"Коэффициент Фишера для данной модели: {f}. Статистика попадает в критическую область, следовательно, "
      f"мы отвергаем гипотезу H0. Различия среднего роста среди взрослых футболистов, хоккеистов "
      f"и штангистов есть")
