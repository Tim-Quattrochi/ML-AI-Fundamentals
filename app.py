from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np


def load_data():
    return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


data = load_data()

print("[Top Rows]: \n", data.head())

print("[Bottom Rows]: \n", data.tail())


missing_data = data.isna()


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html#pandas.DataFrame.dropna
# inplace: Whether to modify the DataFrame rather than creating a new one. Default is False.
# data.dropna(inplace=True)

# Get rid of missing values.

if missing_data.sum().sum() > 0:
    print("There are missing values in the data")
    data.dropna(inplace=True)
    print("Missing values have been removed")


# encode the categorical data
encoded_data = pd.get_dummies(
    data, columns=['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance'])

print("[Encoded Data]: \n", encoded_data.head())


X = data.drop('Attrition', axis=1)
y = data['Attrition']


split_idx = int(0.8 * len(data))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
