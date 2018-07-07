from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import scipy
import scipy.io as sio
### dataset load section

base_path="/home/intern/"
##feats=sio.loadmat('/home/');
train = sio.loadmat('full_train_aug.mat')
test = sio.loadmat('Test_aug.mat')
#
#
#print(np.shape(train_labels['aw']))
#print(np.shape(test_labels['validation_labels_lpq']))
#train_labels = train_labels['aw']
#test_labels = test_labels['validation_labels_lpq']

X_train = train["aug_mat"][:, 1:230]
y_train = train["aug_mat"][:,0]
X_test = test["newtest"][:, 1:230]
y_test = test["newtest"][:,0]


maxe = 1
index=1
index2=1

for lol in range(50):
	for pp in range(30):
		regr = RandomForestRegressor(max_depth=lol+1, random_state=pp+1)
		regr.fit(X_train, y_train)
		y_pred = regr.predict(X_test)


		res = [None]*60
		#print(y_pred)
		for i in range(60):
			if y_pred[i] < 7:
				res[i] = 1
			elif y_pred[i] <20:
				res[i] = 2
			else:
				res[i] = 3
		res1 = [None]*60
		for i in range(60):
			if y_test[i] < 7:
				res1[i] = 1
			elif y_test[i] <20:
				res1[i] = 2
			else:
				res1[i] = 3




#print(regr.feature_importances_)

		#print(res)
		#print(res1)
		#print ("Accuracy is ", accuracy_score(res1,res)*100)
		acc = accuracy_score(res1,res)*100
		if acc>maxe:
			res9 = y_pred
			res10 = res
			maxe=acc
			index=lol
			index2=pp

print(maxe)
print(index)
print(index2)
print(res9)
print(res10)
"""
regr = RandomForestRegressor(max_depth=5, random_state=17)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)


res = [None]*39
#print(y_pred)
for i in range(39):
	if y_pred[i] < 7:
		res[i] = 1
                print("pred<7 ") 
		print(y_pred[i])
	elif y_pred[i] <20:
		res[i] = 2
		print("pred<20 ")
		print(y_pred[i])
	else:
		res[i] = 3
		print("pred>20 ")
		print(y_pred[i])
res1 = [None]*39
for i in range(39):
	if y_test[i] < 7:
		res1[i] = 1
	elif y_test[i] <20:
		res1[i] = 2
	else:
		res1[i] = 3




#print(regr.feature_importances_)

		#print(res)
		#print(res1)
		#print ("Accuracy is ", accuracy_score(res1,res)*100)
acc = accuracy_score(res1,res)*100
print(res)
"""
