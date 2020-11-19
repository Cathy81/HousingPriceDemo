#from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
from dataprepHousing import getData

import os, sys


full_path = os.path.realpath(__file__)
file = os.path.dirname(full_path) + "\\\data\\housingSample.csv"
(X,Y,records)=getData(file)
X_train, X_test, price_train, price_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
model=BayesianRidge()
model.fit(X_train, price_train.ravel())
predPrices=model.predict(X_train)
print(model)
# Summarize the fit of the model

#print(model.intercept_, model.coef_, mse)
print(model.score(X_train, price_train))

predPrices=model.predict(X_train)
mse=mean_squared_error(price_train, predPrices)
rs=r2_score(price_train, predPrices)

print("training mse:",mse)
print("training score:",rs)

# testing
testing_pred_price_results=model.predict(X_test)
mse=mean_squared_error(price_test, testing_pred_price_results)
rs=r2_score(price_test, testing_pred_price_results)
print("median_house_value"+" Predicted_median_house_value")
print(np.c_[price_test, testing_pred_price_results])
print("testing mse:", mse)
print("testing score:",rs)

