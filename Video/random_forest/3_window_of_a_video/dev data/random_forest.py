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
#
#
#print(np.shape(train_labels['aw']))
#print(np.shape(test_labels['validation_labels_lpq']))
#train_labels = train_labels['aw']
#test_labels = test_labels['validation_labels_lpq']

X_train = train["train_data"][0:312, 1:231]
y_train = train["train_data"][0:312,0]
X_test = train["train_data"][312:492, 1:231]
y_test = train["train_data"][312:492,0]


maxe = 1
index=1
index2=1

for lol in range(7):
	for pp in range(7):
		regr = RandomForestRegressor(max_depth=lol+1, random_state=pp+1)
		regr.fit(X_train, y_train)
		y_pred = regr.predict(X_test)

		res3 = [None]*60
		res4 = [None]*60		
		
		for cc in range(60):
			sum = 0
			for k in range(3):
				if(sum<y_pred[(cc)*3 + k]):
					sum = y_pred[(cc)*3 + k];
				
			res3[cc] = sum
			res4[cc] = y_test[(cc)*3]
		res = [None]*60
		#print(y_pred)
		for i in range(60):
			if res3[i] < 7:
				res[i] = 1
			elif res3[i] <20:
				res[i] = 2
			else:
				res[i] = 3

		res1 = [None]*60
		for i in range(60):
			if res4[i] < 7:
				res1[i] = 1
			elif res4[i] <20:
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
			output1 = res3
			maxe=acc
			index=lol
			index2=pp

print(maxe)
print(index)
print(index2)
print(output)
print(output1)


