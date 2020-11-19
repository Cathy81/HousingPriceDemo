#from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
from dataprepHousing import getData
from sklearn.metrics import confusion_matrix
import os, sys
from sklearn.metrics import accuracy_score

full_path = os.path.realpath(__file__)
file = os.path.dirname(full_path) + "\\\data\\housingSample.csv"
(X,Y,records)=getData(file)
X_train, X_test, price_train, price_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
model=BayesianRidge()
model.fit(X_train, price_train.ravel())
predPrices=model.predict(X_train)
# print("neg:", pred_result[0:POSREVIEWSTARTINGPOS])
# print("pos:", pred_result[POSREVIEWSTARTINGPOS:])
# print ("confusion_matrix")
# print(confusion_matrix(Y, pred_result))
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

# #eval.
# test_dir=os.path.dirname(full_path)+"\\dataStudents\\testing"
# test_matrix= extract_features(test_dir, wordCommonDic)
#
# # # to compute the error rate
# pred_result=model.predict(test_matrix)
# files = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]
# for i,file in enumerate(files):
#     print(str(file)+":"+str(pred_result[i]))
# print("Negative reviews in Testing:", pred_result[0:POSTESTING])
# print("Positive reviews in Testing:", pred_result[POSTESTING:])
# print ("confusion_matrix")
# test_labels=np.zeros(TOTALTESTING)
# test_labels[POSTESTING:]=1
# print(confusion_matrix(test_labels, pred_result))
# print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(test_labels, pred_result))
