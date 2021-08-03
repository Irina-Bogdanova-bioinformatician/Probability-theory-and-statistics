import combinatorial_formulas as cf

""" Семь человек рассаживаются наудачу на скамейке. Какова вероятность того, 
    что два определённых человека будут сидеть рядом?
   
    Посчитаем общее количество перестановок среди 7 людей, а также количество перестановок в ситуации, 
    когда двое конкретных людей сидят рядом (считаем этих двоих "одним человеком", 
    умножаем полученное значение перестановок 6 человек на 2, так как люди в паре могут сесть на лавку
    двумя разными способами). Найдем вероятность того, что два определённых человека 
    будут сидеть рядом, разделив второе значение на первое.
"""

a = cf.permutations(6) * 2
print("Количество перестановок в ситуации, когда двое конкретных людей сидят рядом:", a)
b = cf.permutations(7)
print("Количество перестановок среди 7 людей:", b)
print(f"Ответ. Вероятность того, что два определённых человека будут сидеть рядом: {a / b}")
