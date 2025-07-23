
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score





data_1 = pd.read_csv(r'replace with dataset path')
data_1=data_1.drop(['UDI','Product ID','Tool wear [min]','TWF'],axis=1)

data_1.head()

data_1.info()

"""## **3.2 Decribing Dataset**"""

data_1.describe(include='all')

"""## **3.3 Checking Missing Value**

missing values in the data frame
"""

data_1.isnull().sum()

"""## **3.4 Checking Data Duplicate**

checking for the duplicates in the data frame
"""

data_1.duplicated().sum()

duplicate = data_1[data_1.duplicated()]

print("Duplicate Rows :")

# Print the resultant Dataframe
duplicate

data_1.head()

"""no duplicates in the dataframe

## **3.5 Feature Selection**

### **3.5.1 Checking Correlation**
"""

corr_matrix = data_1.drop('Type', axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='rocket', annot=True, linewidths=0.5)
plt.show()


print(corr_matrix['Machine failure'].sort_values(ascending=False))

"""### **3.5.2 Dropping the Feature**

Drop 'Product ID'

Drop variables machine failure, 'TWF','HDF','PWF','OSF',and 'RNF'
"""

data_1 = data_1.drop(['HDF','PWF','OSF','RNF'], axis=1)

"""## **3.6 Data Encoding**

Melakukan data encoding untuk variabel 'Type'.
"""

data_1['Type'][data_1['Type']=='L']=1
data_1['Type'][data_1['Type']=='M']=2
data_1['Type'][data_1['Type']=='H']=3

data_1.head()

"""## **3.7 Checking & Handling Data Imbalance**"""

print(data_1['Machine failure'].value_counts())

pd.Series(data_1['Machine failure']).value_counts().plot(kind='bar', title='Class distribution before appying SMOTE', xlabel='machine failure')

"""Melakukan SMOTE (Synthetic Minority Over-sampling Technique) untuk menyeimbangkan kelas dalam dataset."""

X = data_1.drop(['Machine failure'], axis=1)
y = data_1['Machine failure']

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X, y = smote.fit_resample(X, y)

print(y.value_counts())

pd.Series(y).value_counts().plot(kind='bar', title='Class distribution after appying SMOTE', xlabel='machine failure')

"""## **3.8 Feature Engineering**"""

data_1['Air temperature [K]'] = data_1['Air temperature [K]'] - 273.15
data_1 = data_1.rename(columns={'Air temperature [K]': 'Air temperature [°C]'})
data_1['Process temperature [K]'] = data_1['Process temperature [K]'] - 273.15
data_1 = data_1.rename(columns={'Process temperature [K]': 'Process temperature [°C]'})

data_1['Power']=data_1['Rotational speed [rpm]']*data_1['Torque [Nm]']

data_1['Temperature difference [°C]'] = data_1['Process temperature [°C]']-data_1['Air temperature [°C]']

data_1['Temperature power [°C]'] = data_1['Temperature difference [°C]']/data_1['Power']

data_1 = data_1[['Air temperature [°C]',
         'Process temperature [°C]',
         'Rotational speed [rpm]',
         'Torque [Nm]',
         'Power',
         'Temperature difference [°C]',
         'Temperature power [°C]',
         'Machine failure',
         'Type'
        ]]

data_1.head()

"""# **4. Modeling**"""

X = data_1.drop('Machine failure', axis=1)
y = data_1['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

"""scaling parameters od data frame."""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

"""Model Fitting

## **4.1 Decision Tree Model**
"""

dt = DecisionTreeClassifier().fit(X_train, y_train)
print(X_train[0,:])
y_pred_dt = dt.predict(X_test)

pd.DataFrame(X_train)

y_train

data_1['predictions'] = pd.DataFrame(y_pred_dt)

data_1 = data_1.dropna(subset=['predictions'])
data_1['predictions'] = data_1['predictions'].astype(int)

# Decision Tree Classification Report
print(classification_report(y_test, y_pred_dt))

# Decision Tree confusion matrix
print(confusion_matrix(y_test, y_pred_dt))

"""## **4.2 Random Forest Model**"""

rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5).fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

X_test[0]

# Define the new data point you want to predict
new_data_point = pd.DataFrame({
    'Air temperature [°C]': [299.0],
    'Process temperature [°C]': [309.4],
    'Rotational speed [rpm]': [1758],
    'Torque [Nm]': [25.7],
    'Power': [1758 * 25.7],
    'Temperature difference [°C]': [309.4 - 299.0],
    'Temperature power [°C]': [(309.4 - 299.0) / (1758 * 25.7)],
    'Type': ['L']
})

# Encode the 'Type' feature to numerical values
le = preprocessing.LabelEncoder()
new_data_point['Type'] = le.fit_transform(new_data_point['Type'])

# Create a NumPy array from the new data point
new_data_np = new_data_point.values

# Make predictions for the new data point
y_pred_rf = rf.predict(new_data_np)

# Print the prediction result
if y_pred_rf[0] == 1:
    print("The machine is predicted to fail.")
else:
    print("The machine is predicted to not fail.")


# Random Forest classification report
# Train your Random Forest model
# ...

# Predict on the test data
y_pred_rf_test = rf.predict(X_test)

# Print the classification report for the test data
print(classification_report(y_test, y_pred_rf_test))


# Train your Random Forest model
# ...

# Make predictions for the test dataset
y_pred_rf = rf.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print(conf_matrix)


"""## **4.3 Model Comparison**"""

from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred_dt))
print("Random Forest's Accuracy: ", metrics.accuracy_score(y_test, y_pred_rf))

"""# **5. Evaluation**

## **5.1 Decision Tree - Hyperparameter Tunning**
"""

params = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_dt = GridSearchCV(dt, param_grid=params, cv=5)
grid_search_dt.fit(X_train, y_train)
best_params_dt = grid_search_dt.best_params_
best_score_dt = grid_search_dt.best_score_
print("Best Parameters (Decision Tree):", best_params_dt)
print("Best Score (Decision Tree):", best_score_dt)
y_pred_dt = grid_search_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy (Decision Tree):", accuracy_dt)

"""## **5.2 Decision Tree - Cross Validation**"""

scores = cross_val_score(rf, X, y, cv=10)
print("Accuracy with cross-validation: %.2f with standard deviation %.2f" % (scores.mean(), scores.std()))

"""## **5.3 Feature Importance**"""

feature_importance = pd.DataFrame({'feature': X.columns, 'importance':dt.feature_importances_})
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
print(feature_importance)
plt.figure(figsize=(12,8))
sns.barplot(x='importance', y='feature', data=feature_importance, palette='rocket')
plt.title("Feature Importance Plot")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

pickle.dump(y_pred_rf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
#!pip uninstall fastapi
