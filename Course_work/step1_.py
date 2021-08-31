import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kde

""" Для статистического анализа из набора данных "Heart failure clinical records dataset" возьмем следующие 
    количественные признаки: возраст, уровень креатининфосфокиназы, фракцию выброса, количество тромбоцитов, 
    сывороточный креатинин, сывороточный натрий. Проведем разведочный анализ, построим гистограммы и 
    графики эмпирических функций плотности, проверим гипотезы о нормальности распределения, проведем 
    корреляционный анализ, построим модель логистической регрессии (оценим влияние признаков с нормальным 
    распределением на смерть пациента).
    
    На этом шаге мы найдем квартили, межквартильное расстояние, выбросы, вычислим выборочные средние, медианы,
    несмещенную оценку дисперсии, среднее квадратическое отклонение, построим гистограммы и графики эмпирических 
    функций плотности.
    
    Начнем с квартилей, межквартильных расстояний, выбросов.
"""

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dataset = dataset.drop_duplicates(["age", "creatinine_phosphokinase", "ejection_fraction"])
quartiles_dict = {}
boxplot_dict = {}
outliers_dict = {}
name_list = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine",
             "serum_sodium"]
ind = 0
selected_params = (dataset["age"], dataset["creatinine_phosphokinase"], dataset["ejection_fraction"],
                   dataset["platelets"], dataset["serum_creatinine"], dataset["serum_sodium"])
for el in selected_params:
    quartiles = np.quantile(el, [0.25, 0.75])
    iqr = quartiles[1] - quartiles[0]
    quartiles_dict[name_list[ind]] = quartiles.tolist(), iqr
    boxplot_range = [quartiles[0] - iqr, quartiles[1] + iqr]
    boxplot_dict[name_list[ind]] = boxplot_range
    outliers = [x for x in el if x < boxplot_range[0] or x > boxplot_range[1]]
    outliers_proportion = len(outliers) / len(el)
    outliers_dict[name_list[ind]] = outliers, outliers_proportion
    ind += 1
print("Первый и третий квартили, а также межквартильное расстояние для каждого признака:\n", quartiles_dict)
print("Диапазон значений, лежащих в пределах boxplot (включая усы) для каждого "
      "признака\n", boxplot_dict)
print("Выбросы и доля выбросов для каждого признака:\n", outliers_dict)
print("Дальнейшие исследования попробуем проводить без исключения выбросов ввиду вероятной важности таких значений и "
      "отсутствия возможности проверки точности сбора данных")

""" Теперь посчитаем выборочные средние, несмещенные оценки дисперсии, несмещенные оценки средних квадратических 
    отклонений, медианы.
"""
dict_mean = {}
dict_st_div = {}
dict_variance = {}
dict_median = {}
ind2 = 0
for el in selected_params:
    dict_mean[name_list[ind2]] = np.mean(el)
    dict_st_div[name_list[ind2]] = np.std(el, ddof=1)
    dict_variance[name_list[ind2]] = np.var(el, ddof=1)
    dict_median[name_list[ind2]] = np.median(el)
    ind2 += 1
print("Среднее арифметическое:\n", dict_mean)
print("Медианы:\n", dict_median)
print("Несмещенные оценки дисперсии:\n", dict_variance)
print("Несмещенные оценки средних квадратических отклонений:\n", dict_st_div)

""" Построим гистограммы и эмпирические функции плотности.
"""

fig = plt.figure()
n = 1
for i, var_name in zip(selected_params, name_list):
    ax = fig.add_subplot(2, 3, n)
    i.hist(bins=12, density=True, ax=ax)
    ax.set_title(var_name + " distribution")
    density = kde.gaussian_kde(i)
    x = np.linspace(np.min(i), np.max(i), 300)
    y = density(x)
    plt.plot(x, y)
    n += 1
fig.tight_layout()
plt.show()
