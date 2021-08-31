import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import kde

""" В предыдущих шагах провели разведочный анализ, построили гистограммы и графики эмпирических 
    функций плотности, проверили распределения на нормальность, используя встроенную 
    функцию stats.normaltest(). Распределение значений по каждому выбранному признаку не является нормальным.
    
    На данном этапе анализа удалим строки, содержащие выбросы и еще раз проверим распределения на нормальность 
    с помощью функции stats.normaltest(), построим гистограммы и графики функций плотности. Пересчитаем еще раз 
    выборочные средние, несмещенные оценки дисперсии, несмещенные оценки средних квадратических отклонений, 
    медианы.
"""

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
dataset = dataset.drop_duplicates(["age", "creatinine_phosphokinase", "ejection_fraction"])
name_list = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine",
             "serum_sodium"]


def remove_outlier(df_in, param_name):
    q1 = df_in[param_name].quantile(0.25)
    q3 = df_in[param_name].quantile(0.75)
    iqr = q3 - q1
    boxplot_range = [q1 - iqr, q3 + iqr]
    changed_data = df_in.loc[(df_in[param_name] > boxplot_range[0]) & (df_in[param_name] < boxplot_range[1])]
    return changed_data


for el in name_list:
    dataset = remove_outlier(dataset, el)       # удаляем выбросы
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

dataset.to_csv('dataset_without_outliers.csv', index=False)

""" Создали файл с данными без выбросов по выбранным признакам. 
    Теперь посчитаем еще раз выборочные средние, несмещенные оценки дисперсии, несмещенные оценки средних 
    квадратических отклонений, медианы.
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

""" Построим гистограммы и эмпирические функции плотности для наглядности.
"""

fig = plt.figure()
n = 1
for i, var_name in zip(selected_params, name_list):
    ax = fig.add_subplot(2, 3, n)
    i.hist(bins=10, density=True, ax=ax)
    ax.set_title(var_name + " distribution")
    density = kde.gaussian_kde(i)
    x = np.linspace(np.min(i), np.max(i), 300)
    y = density(x)
    plt.plot(x, y)
    n += 1
fig.tight_layout()
plt.show()
