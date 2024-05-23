from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def load_data():
    return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


data = load_data()


df = pd.DataFrame(data)

print(f"Employee data : \n{df}")

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


categorical_vars = ['BusinessTravel', 'Department',
                    'JobRole', 'MaritalStatus', 'EducationField', 'WorkLifeBalance', 'DailyRate', 'MonthlyIncome', 'RelationshipSatisfaction', 'JobSatisfaction']


categorical_data = data[categorical_vars]

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(categorical_data)


encoded_df = pd.DataFrame(
    encoded_data, columns=encoder.get_feature_names_out(categorical_vars))

df_final = pd.concat([data.drop(columns=categorical_vars), encoded_df], axis=1)


print("[One Hot Encoded Data]: \n", df_final)

#### Creating new features ####
df_final['YearsAtCompanyPerRole'] = df_final['YearsAtCompany'] / \
    (df_final['TotalWorkingYears'] + 1)  # Adding 1 to avoid division by zero


df_final['AgeOverYearsAtCompany'] = df_final['Age'] / \
    (df_final['YearsAtCompany'] + 1)  # add 1 to avoid division by zero


df_final['LeaveBefore5Years'] = (df_final['YearsAtCompany'] < 5).astype(int)


df_final['StealFromCompany'] = (
    (df_final['Age'] < 20) |
    (df_final['YearsAtCompany'] < 1) |
    (df_final['YearsSinceLastPromotion'] == 0) |
    (df_final['Gender'] == 'Male')
).astype(int)


print("[Data with New Features]: \n", df_final.head())


numerical_vars = df_final.select_dtypes(
    include=['int64', 'float64']).columns.tolist()


if 'Attrition' in numerical_vars:
    numerical_vars.remove('Attrition')


scaler = StandardScaler()
df_final[numerical_vars] = scaler.fit_transform(df_final[numerical_vars])

print("[Standardized Data]: \n", df_final.head())


################### This section preprocesses the data, trains a logistic regression model, and evaluates its performance on predicting employee attrition. ##############

data = pd.get_dummies(data)


scaler = StandardScaler()
numerical_features = ['Age', 'DailyRate', 'DistanceFromHome',
                      'HourlyRate', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

X = data.drop(columns=['Attrition_Yes', 'Attrition_No'])
y = data['Attrition_Yes']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

feature_importance = pd.DataFrame(
    {'feature': X.columns, 'importance': logreg.coef_[0]})


feature_importance['abs_importance'] = feature_importance['importance'].abs()
feature_importance = feature_importance.sort_values(
    by='abs_importance', ascending=False)

print(feature_importance)


y_pred = logreg.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


##### Fine Tuning the Model ##########

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500, 1000]
}

grid_search = GridSearchCV(
    LogisticRegression(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print("Best Parameters:", best_params)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Best Model Precision:", precision_score(y_test, y_pred_best))
print("Best Model Recall:", recall_score(y_test, y_pred_best))
print("Best Model F1 Score:", f1_score(y_test, y_pred_best))
print("Best Model Classification Report:\n",
      classification_report(y_test, y_pred_best))


##### predictions ####

y_pred_logreg = logreg.predict(X_test)


print("Logistic Regression Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Precision:", precision_score(y_test, y_pred_logreg))
print("Recall:", recall_score(y_test, y_pred_logreg))
print("F1 Score:", f1_score(y_test, y_pred_logreg))
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))
