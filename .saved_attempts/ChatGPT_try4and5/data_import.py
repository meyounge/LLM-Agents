import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv('./Train_Test_Data/dataset.csv')
print('Data loaded successfully')