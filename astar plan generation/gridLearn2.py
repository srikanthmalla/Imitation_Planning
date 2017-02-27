#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD , Adam
import numpy as np
import random
import collections
import pandas

def modifyData(data):
	"""Data is (x,106) 
	Need to modify 100th and 101th term in array base 0
	"""
	mod_data = np.zeros((data.shape[0],data.shape[1] - 2 + 14))

	for i in range(data.shape[0]):
		start = '{0:07b}'.format(data[i,100])
		print start, data[i,100]
		end = '{0:07b}'.format(data[i,101])
		mod_data[i,100:106]=start
		mod_data[i,107:113]=end
	return mod_data



#Network Design
model = Sequential()
learning_rate=1e-4
epochs=200
no_inputs=102
model.add(Dense(114,init='uniform',input_shape=(no_inputs,),activation='relu'))
# model.add(Dense(114,init = 'uniform', activation='relu'))
model.add(Dense(4,init = 'uniform', activation = 'softmax'))

adam = Adam(lr = learning_rate)


model.compile(loss = 'binary_crossentropy',optimizer='Adam')


#Data Modelling
data_input = pandas.read_excel(open('data/input2.xls','rb'))
data_output = pandas.read_excel(open('data/output2.xls','rb'))

data_input1 = np.array(data_input[0:6925])
data_output1 = np.array(data_output[0:6925])

print "Input Data Shape: ",data_input1.shape
print "Output Data Shape: ",data_output1.shape

complete_data = np.concatenate((data_input1,data_output1), axis=1)


# This is to Avoid data coreelation
np.random.shuffle(complete_data)

data_input1 = complete_data[:,0:102]
# data_input1 = modifyData(complete_data[:,0:102])
data_output1 = complete_data[:,102:106]

# print np.shape(data_input1), np.shape(data_output1)
#Training Data
train_x = data_input1[0:4000]
train_y = data_output1[0:4000]


train_x = np.array(train_x)
train_y = np.array(train_y)
# Machine Learning
#model.train_on_batch(train_x, train_y)
model.fit(train_x, train_y, nb_epoch = epochs, batch_size = 32)
#Testing data
test_x = data_input1[4000:6925]
print "Test Input data shape: ", test_x.shape
test_y = data_output1[4000:6925]
print "Test Output data shape: ", test_y.shape
test_x = np.array(test_x)
test_y = np.array(test_y)

predict_y = model.predict(np.array(test_x), batch_size=1)

no_match = 0
# np.shape(predict_y)
for i in range(2000):
	# print predict_y[i,:]
	# print np.argmax(predict_y,axis=0)
	if np.argmax(predict_y[i,:],axis=0) != np.argmax(test_y[i,:], axis = 0):
		no_match = no_match + 1
	#print "Predicted action: ", np.argmax(predict_y[i,:],axis=0)
	#print "Actual action: ",np.argmax(test_y[i,:], axis = 0)
	#print "-----------------------------------------------------"

print no_match

#Predict
#newstate = GetState()
#model.predict()