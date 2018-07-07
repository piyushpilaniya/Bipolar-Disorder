import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import scipy
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression
train = sio.loadmat('Train.mat')
test = sio.loadmat('Dev.mat')
X_train = train["final1"][:, 1:620]
y_train = train["final1"][:,0]
X_test = test["final"][:, 1:620]
y_test = test["final"][:,0]
pls2 = PLSRegression(n_components=11)
pls2.fit(X_train, y_train)

XX = [[1,2,2], [2,3,4]]

y_pred = pls2.predict(X_test)

res = [None]*54
for i in range(54):
	if y_pred[i] < 7:
		res[i] = 1
	elif y_pred[i] <20:
		res[i] = 2
	else:
		res[i] = 3


print(res)
#print ("Accuracy is ", accuracy_score(y_test,res)*100)
