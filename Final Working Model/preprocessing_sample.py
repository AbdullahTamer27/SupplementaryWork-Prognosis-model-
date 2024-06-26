import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Load data
file_path = '../input/myocardial-infarction-complications/Myocardial infarction complications Database.csv'

data = pd.read_csv(file_path)
data.head()

data.drop(data.columns[[0, 92, 93, 94, 99, 100, 101, 102, 103, 104]], axis=1, inplace=True)

missing_val_count_by_column = (data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

nan_cols = [i for i in data.columns if data[i].isnull().any()]

nan_cols85 = [i for i in data.columns if data[i].isnull().sum() > 0.85*len(data)]

print(nan_cols85)

data_updated = data.drop(columns=nan_cols85)

imputer = IterativeImputer(random_state=42)

imputed = imputer.fit_transform(data_updated)

data_imputed = pd.DataFrame(imputed, columns=data_updated.columns)

data_imputed.to_csv('post_imputation_data.csv')
data = data_imputed.apply(pd.to_numeric, errors='coerce')
data = data.rename(columns={'LET_IS_0.0': 'alive', 
			    'LET_IS_1.0': 'cardiogenic shock',
                            'LET_IS_2.0': 'pulmonary edema', 
                            'LET_IS_3.0': 'myocardial rupture',
                            'LET_IS_4.0': 'progress of congestive heart failure', 
                            'LET_IS_5.0': 'thromboembolism', 
                            'LET_IS_6.0':'asystole', 
                            'LET_IS_7.0':'ventricular fibrillation'})

