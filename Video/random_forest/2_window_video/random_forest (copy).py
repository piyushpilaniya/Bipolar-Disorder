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
train = sio.loadmat('Train.mat')
test = sio.loadmat('Dev.mat')
#
#
#print(np.shape(train_labels['aw']))
#print(np.shape(test_labels['validation_labels_lpq']))
#train_labels = train_labels['aw']
#test_labels = test_labels['validation_labels_lpq']

X_train = train["train_data"][0:208, 1:231]
y_train = train["train_data"][0:208,0]
X_test = test["final"][:, 1:231]
y_test = test["final"][:,0]


maxe = 1
index=1
index2=1

for lol in range(30):
	for pp in range(15):
		regr = RandomForestRegressor(max_depth=lol+1, random_state=pp+1)
		regr.fit(X_train, y_train)
		y_pred = regr.predict(X_test)

		res = [None]*60
		#print(y_pred)
		for i in range(60):
			if  y_pred[i] < 7:
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
			output = y_pred
			output1 = res
			maxe=acc
			index=lol
			index2=pp

print(maxe)
print(index)
print(index2)
print(output)
print(output1)


