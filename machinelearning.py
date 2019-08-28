import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from math import sqrt
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_excel('C:/project/credit.xlsx')
print(df.shape)
print(df)
print(df.describe())
boxplot = df.boxplot(column=['Balance'])

pairplot = pd.plotting.scatter_matrix(df, figsize=(7, 7), marker='o', hist_kwds={'bins': 7}, s=60, alpha=0.8)
plt.show()

corr = df.corr()
print(corr)

features = df.iloc[:, 0:6]
print(features)
labels = df.iloc[:, -1]
print(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
regr.score(X_test, y_test)
y_pred = regr.predict(X_test)

print('Intercept\t\t:', regr.intercept_)
print('Coefficients\t:', regr.coef_)
print('Test Score\t\t:', regr.score(X_test, y_test))
print("\n")

print("Balance = {:.5} + {:.5}*Income + {:.5}*Limit + {:.5}*Rating + {:.5}*Cards + {:.5}*Age + {:.5}*Education ".format(regr.intercept_, regr.coef_[0], regr.coef_[1], regr.coef_[2], regr.coef_[3], regr.coef_[4], regr.coef_[5]))
print("\n")

x = df.Limit
y = df.Rating
plt.scatter(x, y)
plt.ylabel("Rating")
plt.xlabel("Limit")
plt.title("Limit-Rating");
plt.grid()
plt.show()

X = add_constant(df)
print(pd.Series([variance_inflation_factor(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns))

x_train_1 = X_train.copy()
x_test_1 = X_test.copy()
x_train_1 = x_train_1.drop(columns="Limit")
x_test_1 = x_test_1.drop(columns="Limit")
regr1 = linear_model.LinearRegression()
regr1.fit(x_train_1, y_train)
regr1.score(x_test_1, y_test)
print('R^2(Drop limit) \t:', regr1.score(x_test_1, y_test))

x_train_2 = X_train.copy()
x_test_2 = X_test.copy()
x_train_2 = x_train_2.drop(columns="Rating")
x_test_2 = x_test_2.drop(columns="Rating")
regr2 = linear_model.LinearRegression()
regr2.fit(x_train_2, y_train)
regr2.score(x_test_2, y_test)
print('R^2(Drop Rating) \t:', regr2.score(x_test_2, y_test))

print('Intercept\t\t:', regr1.intercept_)
print('Coefficients\t:', regr1.coef_)
print('Test Score\t:', regr1.score(x_test_1, y_test))
y_pred = regr1.predict(x_test_1)
print(y_pred)

#standardize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_std = sc.fit_transform(x_train_1)
X_test_std = sc.transform(x_test_1)

lr = linear_model.LinearRegression()
lr.fit(X_train_std, y_train)
y_train_pred = lr.predict(X_train_std)
y_test_pred = lr.predict(X_test_std)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test, y_test_pred)))


comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

comparison.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print("The first model is: Balance = {:.5} + {:.5}*Income + {:.5}*Limit + {:.5}*Rating + {:.5}*Cards + {:.5}*Age + {:.5}*Education ".format(regr.intercept_, regr.coef_[0], regr.coef_[1], regr.coef_[2], regr.coef_[3], regr.coef_[4], regr.coef_[5]))
print("The final model is: Balance = {:.5} + {:.5}*Income + {:.5}*Rating + {:.5}*Cards + {:.5}*Age + {:.5}*Education ".format(lr.intercept_, lr.coef_[0], lr.coef_[1], lr.coef_[2], lr.coef_[3], lr.coef_[4]))

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(lr.score(X_test_std, y_test)))


# model evaluation for training set
rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
r2 = r2_score(y_train, y_train_pred)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
r2 = r2_score(y_test, y_test_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))