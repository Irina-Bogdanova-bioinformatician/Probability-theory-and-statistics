import pandas as pd
import itertools
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

""" В предыдущих шагах провели разведочный анализ, построили гистограммы и графики эмпирических 
    функций плотности, проверили распределения на нормальность, используя встроенную 
    функцию stats.normaltest(). После удаления выбросов распределения значений по следующим признакам 
    оказались нормальными: возраст, фракция выброса, количество тромбоцитов, сывороточный натрий. Далее будем 
    работать только с этими признаками.

    На данном этапе анализа проведем корреляционный анализ, построим модель логистической регрессии.
"""

dataset = pd.read_csv('dataset_without_outliers.csv')
pd.set_option('display.max_columns', None)
print("Коэффициенты корреляции Пирсона:\n", dataset[["age", "ejection_fraction", "platelets", "serum_sodium",
                                                     "DEATH_EVENT"]].corr())
print("Как видно из полученных данных, уровень линейной зависимости между признаками, оцененный с помощью "
      "коэффициентов корреляции Пирсона, во всех случаях довольно низкий")

""" Проверим гипотезу о значимости корреляции.
    H0 - корреляция незначима (коэффициент корреляции равен 0)
"""
corr_estimation_dict = {}   # ключи - названия признаков, значения - t для корреляции этих признаков и кр область
sign_corr_dict = {}   # ключи - названия признаков, значения - t для корреляции этих признаков и кр область
name_list = ["age", "ejection_fraction", "platelets", "serum_sodium", "DEATH_EVENT"]
comb_of_features = list(itertools.combinations(name_list, 2))     # лист комбинаций по 2 из названий признаков
n = dataset.shape[0]
alpha = 0.05
t1 = stats.t.ppf(alpha / 2, df=n - 2)
t2 = stats.t.ppf(1 - alpha / 2, df=n - 2)
for i in comb_of_features:
    corr = dataset[i[0]].corr(dataset[i[1]])
    t = (corr * np.sqrt(n - 2)) / np.sqrt(1 - corr ** 2)
    corr_estimation_dict[i] = t
    if t < t1 or t > t2:
        sign_corr_dict[i] = t
print(f"Критическая область: (-∞, {t1})u({t2}, +∞)")
print("Значения статистики t для коэффициентов корреляции признаков:\n",
      corr_estimation_dict)
print("Коэффициент корреляции значим для следующих сочетаний признаков (показан t для коэффициентов корреляции):\n",
      sign_corr_dict)

""" Корреляционный анализ показал значимость корреляции между признаками 'age', 'DEATH_EVENT' и между признаками
    'ejection_fraction', 'DEATH_EVENT'
    
    Теперь построим модель логистической регрессии для этих признаков. Посмотрим, как возраст и фракция выброса
    влияют на бинарную переменную 'DEATH_EVENT'. Используем библиотеки sklearn и statsmodels.
"""

model = LogisticRegression(solver='liblinear', fit_intercept=True)
model.fit(dataset[["age", "ejection_fraction"]], dataset["DEATH_EVENT"])
b_sklearn = model.coef_.flatten()
print(f"Коэффициенты модели логистической регрессии:{b_sklearn}")

log_model = sm.Logit(dataset["DEATH_EVENT"], dataset[["age", "ejection_fraction"]])
result = log_model.fit()
print("Результаты, полученные с помощью библтотеки statsmodels\n", result.summary2())

""" Из полученных данных (таблица Logit) видно, что полученные коэффициенты логистической регрессии значимы 
   (статистики z для обоих коэффициентов принадлежат критической области).
"""
