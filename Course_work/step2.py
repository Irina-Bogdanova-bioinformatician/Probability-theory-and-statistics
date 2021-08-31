import pandas as pd
import numpy as np
from scipy import stats

""" Для статистического анализа из набора данных "Heart failure clinical records dataset" взяли следующие 
    количественные признаки: возраст, уровень креатининфосфокиназы, фракцию выброса, количество тромбоцитов, 
    сывороточный креатинин, сывороточный натрий. 
    
    В предыдущем шаге провели разведочный анализ, построили гистограммы и графики эмпирических 
    функций плотности. Требуется дальнейшая проверка распределений на нормальность. Используем встроенную 
    функцию stats.normaltest().
"""

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dataset = dataset.drop_duplicates(["age", "creatinine_phosphokinase", "ejection_fraction"])
name_list = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine",
             "serum_sodium"]
selected_params = (dataset["age"], dataset["creatinine_phosphokinase"], dataset["ejection_fraction"],
                   dataset["platelets"], dataset["serum_creatinine"], dataset["serum_sodium"])
ind = 0
for el in selected_params:
    loc = np.mean(el)
    scale = np.std(el, ddof=1)
    statistic, p = stats.normaltest(el)
    alpha = 0.05
    print(f"Результаты теста на нормальность для признака {name_list[ind]}:\n statistic = {statistic}, p-value = "
          f"{p}")
    if p < alpha:
        print(f"Нулевая гипотеза может быть отклонена - распределение значений признака {name_list[ind]} "
              f"отличается от нормального")
    else:
        print(f"Принимаем нулевую гипотезу - распределение значений признака {name_list[ind]} нормальное")
    ind += 1

""" Как видно из полученных данных, распределение значений по каждому выбранному признаку не является нормальным.
    В шаге 3 удалим выбросы и еще раз проверим распределения на нормальность.
"""
