# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 00:08:01 2018

@author: intern
"""

from __future__ import print_function
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, SpatialDropout2D, Lambda
from keras.layers import advanced_activations
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
#from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2
#import theano.gpuarray
#theano.gpuarray.use("cuda")

import h5py as h5
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

#from keras.layers.core import  Lambda
#from keras.layers import  Merge
import sklearn
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.regularizers import l1_l2, Regularizer
from keras.engine import Layer
from theano import function, shared, printing
from keras.engine import InputSpec
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation,     Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import theano.tensor as T
import scipy.io as sio
from keras.layers import LSTM
#%%
from keras.utils.generic_utils import get_custom_objects
#
from sklearn.cross_validation import train_test_split
import keras

from keras.layers.merge import add , multiply, Dot, Multiply

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

batch_size = 100
SEQUENCES_PER_VIDEO=100;
FEATURES_PER_SEQUENCE=177;
FEATURES_PER_SEQUENCE_LPQ=768;
FEATURES_PER_SEQUENCE_HNE=9;
class SortLayer(Layer):
    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
    def __init__(self,k=1,label=1,**kwargs):
        # k is the factor we force to be 1
        self.k = k*1.0
        self.label = label
        
        self.input_spec = [InputSpec(ndim=4)]
        
        super(SortLayer, self).__init__(**kwargs)

    def build(self,input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x,mask=None):
        
        x=K.reshape(x, (-1, SEQUENCES_PER_VIDEO))

#        x2=[0,1,2,3];
#        response=multiply([x, x2])
        response =K.mean(x, axis=1, keepdims=False)

        #response =K.mean(x, axis=1, keepdims=False)
        return response

    

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], 1])
        #return input_shape

    def compute_output_shape(self, input_shape):
       # print(input_shape)
        return tuple([input_shape[0], 1])


def createModel2():
    model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(32, input_shape=(799, 230),return_sequences=False))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))

#model.add(Flatten())
    model.add(Dense(4, activation='sigmoid'))
   # Lambda(lambda x: x * 2)
    model.add(SortLayer())

    return model


# In[7]:


##labels
labels_base_path="/home/intern/internship_avec/Dev_geometric/window_3d/"
#feats=sio.loadmat('/home/amanjot/aman/segments_dataset/');
train_labels = sio.loadmat(labels_base_path + 'output_train_ymrs.mat')
test_labels = sio.loadmat(labels_base_path + 'output_dev_ymrs.mat')

one_hot_labels1 = keras.utils.to_categorical(train_labels, num_classes=3)
one_hot_labels2 = keras.utils.to_categorical(test_labels, num_classes=3)

print(np.shape(train_labels['aw']))
print(np.shape(test_labels['validation_labels_lpq']))
train_labels = train_labels['aw']
test_labels = test_labels['validation_labels_lpq']
test_labels=test_labels/3
train_labels=train_labels/3
###LBPTOP
base_path="/home/intern/internship_avec/Dev_geometric/window_3d/"
train_features=sio.loadmat(base_path+'output_train.mat')
test_features=sio.loadmat(base_path+'output_dev.mat')
train_features=train_features['a']
test_features=test_features['a']
#########
print(np.shape(train_features))
print(np.shape(test_features))
#
#######LPQTOP
#base_path="/home/amanjot/aman/segments_dataset/1500/LPQTOP/"
#train_features_l=sio.loadmat(base_path+'train_segments_LPQTOP_1500.mat')
#test_features_l=sio.loadmat(base_path+'validation_segments_LPQTOP_1500.mat')
#train_features_l=train_features_l['a']
#test_features_l=test_features_l['a']
##########
#print(np.shape(train_features_l))
#print(np.shape(test_features_l))
#base_path="/home/amanjot/aman/segments_dataset/1500/HeadPose/"
#train_features_h=sio.loadmat(base_path+'train.mat')
#test_features_h=sio.loadmat(base_path+'validation.mat')
#train_features_h=train_features_h['ans']
#test_features_h=test_features_h['a']
#print(np.shape(train_features_h))
#print(np.shape(test_features_h))
#
#
####Augmented Head Pose
#base_path="/home/amanjot/aman/segments_dataset/1500/Augmented/"
#train_features_h=sio.loadmat(base_path+'Train_HP.mat')
#train_features_h=train_features_h['f_HP']
#train_features_h=train_features_h[:,:,0:9]
#
#train_labels = sio.loadmat(base_path+'Labels_Train.mat')
#test_labels = sio.loadmat(base_path + 'lB_VAL.mat')
#train_labels = train_labels['f_lb']
#test_labels = test_labels['labels_validation']
#train_labels=train_labels/3
#test_labels=test_labels/3
#print(np.shape(train_features_h))
#
##########
#
####Augmented Head Pose
#base_path="/home/amanjot/aman/final_models/dataset/"
#train_features_h=sio.loadmat(base_path+'Augmented_Train_frame_Head_1500.mat')
#train_features_h=train_features_h['frame']
#train_features_h=train_features_h[:,:,0:9]
#train_labels = sio.loadmat(base_path+'Labels_Train.mat')
#test_labels = sio.loadmat(base_path + 'lB_VAL.mat')
#train_labels = train_labels['f_lb']
#test_labels = test_labels['labels_validation']
#train_labels=train_labels;
#test_labels=test_labels;
#train_labels=train_labels/3
#test_labels=test_labels/3
#test_features_n=sio.loadmat(base_path+'val_frame_Head_1500.mat')
#test_features_n=test_features_n['a2']
#print(np.shape(train_features_h))
## In[8]:


import random


learning_rate=6e-5
sgd = Adam(lr=learning_rate) #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


#nb_epoch = 10
#batch_size = 256


# fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)
#
#
#from keras import callbacks
#
#final_score=[];
#
#for i in range(1):
#    i=i+1;
#    print(i)
#    seed = random.randint(1,101)
#    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
#    for train_index, val_index in sss.split(train_features_h, train_labels):
#        print("ITERATION:" + str(i) + " TRAIN:", train_index, "VAL:", val_index)
#        x_train, x_val = train_features_h[train_index], train_features_h[val_index]
#        y_train, y_val = train_labels[train_index], train_labels[val_index]
#        
##    for train_index, val_index in sss.split(train_features_lpq, train_labels):
##        print("LPQ ITERATION:" + str(i) + " TRAIN:", train_index, "VAL:", val_index)
##        x_train_lpq, x_val_lpq = train_features_lpq[train_index], train_features_lpq[val_index]
#
#    y_train=y_train.reshape((np.shape(y_train)[0],1))
#    y_val=y_val.reshape((np.shape(y_val)[0],1))
#
#    print(np.shape(x_train))
#    print(np.shape(x_val))
#    #print(np.shape(x_train_lpq))
#    #print(np.shape(x_val_lpq))
#    print(np.shape(y_train))
#    print(np.shape(y_val))
#   
   
    model=createModel2()
    print(model.summary())
    # Compile model

    model.compile(loss='mse', optimizer=sgd, metrics=['mse']) 

    m_name='prediction_LSTM_without_augmentation_new'+str(i)+'.h5'
#    
    checkpoint = callbacks.ModelCheckpoint(m_name, monitor='val_mean_squared_error',
                                          save_best_only=True, save_weights_only=False, verbose=1,mode='min')
    early_stopping=callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=200, verbose=0, mode='min')
    #print(np.shape(x_val_lpq))
    print(np.shape(y_val))
    #class_weight = {0:50,0.33333333333333331:6,0.66666666666666663:2,1:9}
    #model.fit([x_train], y_train, epochs=20,batch_size=1,verbose=1,
                 # validation_data=([x_val],y_val),class_weight=class_weight)
   # model.fit([x_train], y_train, epochs=20,batch_size=1,verbose=1,
                  #validation_data=([x_val],y_val),callbacks=[checkpoint,early_stopping])
    model.fit([x_train], y_train, epochs=40,batch_size=1,verbose=1,
                  validation_data=([x_val],y_val),callbacks=[checkpoint,early_stopping])
    
   # model.load_weights(m_name)
    print("%%%%%%%%%%%%%%%%")
    #print(np.shape(x_test))
    print(np.shape(test_features_h))
    print("%%%%%%%%%%%%%%%%")

    #score = model.evaluate([x_test,x_test_lpq], y_test,  verbose=0)
    #model.save('DeepMILMaxPooling_swish.h5') 
    
    prediction1 = model.predict([test_features_n[:,:,0:9]], batch_size=1)
    np.shape(prediction1)
    
    error1=sklearn.metrics.mean_squared_error(test_labels, prediction1)
    print(error1)
    #sio.savemat('prediction_aamir_deep_mil_lbptop.mat', {'prediction' : prediction})
    final_score.append(error1)
    #all_pred= model.predict(X, batch_size=1)
    #score = model.evaluate(X, Y_Ori,  verbose=0)
    #sio.savemat('prediction_deepmil_swish_BestModel.mat', {'prediction' : all_pred, 'Original_labels': Y_Ori})
    
#sio.savemat('prediction_deepmil_swish_MaxPooling_only13.mat', {'prediction' : prediction, 'Original_labels': Y_o})
print ("All Iter Scores:" + str(final_score))
print(np.mean(final_score))
#sio.savemat('/home/amanjot/Desktop/aman_deepMil/final/prediction_LSTM_augmented_HP_new_roll.mat', {'prediction' : prediction, 'Original_labels': test_labels})
#layer_name='dense_60'; #name of the layer to pick
#model.load_weights(m_name)
#
#  
#intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
#
#
#intermediate_output_train = intermediate_layer_model.predict(train_features_h, batch_size=1, verbose=1)
##intermediate_output_val = intermediate_layer_model.predict(x_val, batch_size=1, verbose=1)
#intermediate_output_test = intermediate_layer_model.predict(test_features_h[:,:,6:9], batch_size=1, verbose=1)
#import  scipy.io
#scipy.io.savemat('intermediate_output_train_HP_roll_std.mat', mdict={'values':intermediate_output_train })    
##scipy.io.savemat('intermediate_output_val_HP.mat', mdict={'values':intermediate_output_val })    
#scipy.io.savemat('intermediate_output_test_HP_roll_std.mat', mdict={'values':intermediate_output_test })  
###predict accuracy
#
# #prediction[test_labels=0]
#  
#
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#
#pca = PCA(n_components=20)
#pca_result = pca.fit_transform(intermediate_output_test)
#print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
###Variance PCA: 0.993621154832802
#
##Run T-SNE on the PCA features.
#tsne = TSNE(n_components=2, verbose = 1,learning_rate=80.0,perplexity=6.0)
#tsne_results = tsne.fit_transform(pca_result)
#from keras.utils import np_utils
#import matplotlib.pyplot as plt
##matplotlib inline
#
#y_test_cat = np_utils.to_categorical(test_labels*3, num_classes = 4)
#color_map = np.argmax(y_test_cat, axis=1)
#plt.figure(figsize=(10,10))
#for cl in range(4):
#    indices = np.where(color_map==cl)
#    indices = indices[0]
#    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
#plt.legend()
#plt.show()
#------------------------------
a=test_labels[[np.where(test_labels==0)]][0]-prediction1[[np.where(test_labels==0)]][0];
s=a[0]*a[0];
d=sum(s)
f0=d/4
#------------------------------------------------------------------
a=test_labels[[np.where(test_labels*3==1)]][0]-prediction1[[np.where(test_labels*3==1)]][0];
s=a[0]*a[0];
d=sum(s)
f1=d/10
#---------------------------------------------------------------
a=test_labels[[np.where(test_labels*3==1)]][0]-prediction[[np.where(test_labels*3==1)]][0];
s=a[0]*a[0];
d=sum(s)
f=d/10
#------------------------------------------------------------------------------
a=test_labels[[np.where(test_labels*3==2)]][0]-prediction1[[np.where(test_labels*3==2)]][0];
s=a[0]*a[0];
d=sum(s)
f2=d/19
#---------------------------------------------------------------------------
a=test_labels[[np.where(test_labels*3==3)]][0]-prediction1[[np.where(test_labels*3==3)]][0];
s=a[0]*a[0];
d=sum(s)
f3=d/15
