import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv("House_Rent_Dataset.csv")
data

data['Posted On'] = pd.to_datetime(data['Posted On'])

Q1=data['BHK'].quantile(0.25)
Q3=data['BHK'].quantile(0.75)
IQR=Q3-Q1
Lower_Bound = Q1 - (1.5 * IQR)
Upper_Bound = Q3 + (1.5 * IQR)

data.loc[(data['BHK']>Upper_Bound)] = data['BHK'].median()

Q1_Rent=data['Rent'].quantile(0.25)
Q3_Rent=data['Rent'].quantile(0.75)
IQR_Rent=Q3_Rent-Q1_Rent
LowerBoundRent = Q1_Rent - 1.5 * IQR_Rent
UpperBoundRent = Q3_Rent + 1.5 * IQR_Rent

data.loc[(data['Rent']>UpperBoundRent)] = UpperBoundRent

Q1_Size=data['Size'].quantile(0.25)
Q3_Size=data['Size'].quantile(0.75)
IQR_Size=Q3_Size-Q1_Size
LowerBoundSize = Q1_Size - 1.5 * IQR_Size
UpperBoundSize = Q3_Size + 1.5 * IQR_Size

data.loc[(data['Size']>UpperBoundSize)] = UpperBoundSize

Q1_Bathroom=data['Bathroom'].quantile(0.25)
Q3_Bathroom=data['Bathroom'].quantile(0.75)
IQR_Bathroom=Q3_Bathroom-Q1_Bathroom
LowerBoundBathroom = Q1_Bathroom - 1.5 * IQR_Bathroom
UpperBoundBathroom = Q3_Bathroom + 1.5 * IQR_Bathroom

data.loc[(data['Bathroom']>UpperBoundBathroom)] = data['Bathroom'].median()

data.drop(['Posted On', 'Floor','Area Locality'],axis=1,inplace=True)

data[['BHK','Rent','Size','Bathroom']] = data[['BHK','Rent','Size','Bathroom']].astype(int)

#Encoding Categorical Data
data2 = data.join(pd.get_dummies(data[['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']], drop_first=True))
data2.drop(columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'], inplace=True)

#Memisahkan training dan testing data
X=data2.drop('Rent', axis=1)
y=data2['Rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Membuat fungsi evaluate untuk evaluasi model berdasarkan MAE, RMSE, dan R2 Score
def evaluate(model):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    print('MAE:', mean_absolute_error(y_test, pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
    print('R2 Score:', r2_score(y_test, pred))

#Evaluasi Linear Regression
evaluate(LinearRegression())

# Create and train the linear regression model
linearmodel = LinearRegression()
linearmodel.fit(X_train, y_train)

# Display coefficients and intercept
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': linearmodel.coef_})
intercept = linearmodel.intercept_
print('\nCoefficients:\n', coefficients)
print('\nIntercept:', intercept)

# Evaluate how well the model fits the training data
print("Linear Regression Training Accuracy:", linearmodel.score(X_train, y_train))

# Evaluate the model on the test set
print("Linear Regression Test Accuracy:", linearmodel.score(X_test, y_test))

from sklearn.linear_model import Lasso

evaluate(Lasso())
modellasso = Lasso(alpha=1.0)
modellasso.fit(X_train, y_train)

# Print the coefficients
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': modellasso.coef_})
print(coefficients)

# Print the intercept
intercept_value = modellasso.intercept_
print(f'Intercept: {intercept_value}')
y_pred = modellasso.predict(X_test)


pickle.dump(modellasso, open("model.pkl", "wb"))  # Corrected mode to write binary