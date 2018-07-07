import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import scipy
import scipy.io as sio
### dataset load section
balance_data = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                           sep= ',', header= None)
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

X_train = train["final1"][:, 1:620]
y_train = train["final1"][:,0]
X_test = test["final"][:, 1:620]
y_test = test["final"][:,0]




#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
y_pred = clf_entropy.fit(X_train, y_train)


y_pred = clf_gini.predict(X_test)
print(y_pred)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
	

