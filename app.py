
import pandas as pd
import numpy as np


def load_data():
    return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


data = load_data()

print(data.head())
