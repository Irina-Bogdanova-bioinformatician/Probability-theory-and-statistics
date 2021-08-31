import pandas as pd

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print(dataset)
