from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD , Adam
import numpy as np
import random
import collections
import pandas

def modifyData(data):
	"""Data is (x,106) 
	Need to modufy 100th and 101th term in array base 0
	"""
	mod_data = np.zeros((data.shape[0],data.shape[1] - 2 + 14))

	for i in range(data.shape[0]):
		start = bin(data[i,100])[2:]
		end = bin(data[i,101])[2:]


#Network Design
model = Sequential()
model.add(Dense(200,init='uniform',input_shape=(102,),activation='relu'))
model.add(Dense(200,init = 'uniform', activation='relu'))
model.add(Dense(4,init = 'uniform', activation = 'sigmoid'))

adam = Adam(lr = 1e-4)


model.compile(loss = 'mse',optimizer=adam)


#Data Modelling
data_input = pandas.read_excel(open('input1.xls','rb'))
data_output = pandas.read_excel(open('output1.xls','rb'))

data_input1 = np.array(data_input[0:6925])
data_output1 = np.array(data_output[0:6925])

print "Input Data Shape: ",data_input1.shape
print "Output Data Shape: ",data_output1.shape

complete_data = np.concatenate((data_input1,data_output1), axis=1)


# This is to Avoid data coreelation
np.random.shuffle(complete_data)

data_input1 = complete_data[:,0:102]
data_output1 = complete_data[:,102:106]



#Training Data
train_x = data_input1[0:4000]
train_y = data_output1[0:4000]

#Testing data
test_x = data_input1[4000:6925]
print "Test Input data shape: ", test_x.shape
test_y = data_output1[4000:6925]
print "Test Output data shape: ", test_y.shape
test_y = np.array(test_y)

train_x = np.array(train_x)
train_y = np.array(train_y)
# Machine Learning
#model.train_on_batch(train_x, train_y)

model.fit(train_x, train_y, nb_epoch = 100, batch_size = 32)

predict_y = model.predict(np.array(test_x), batch_size=1)

no_match = 0

for i in range(2000):
	#print predict_y[i,:]
	#print test_y[i,:]
	if np.argmax(predict_y[i,:],axis=0) != np.argmax(test_y[i,:], axis = 0):
		no_match = no_match + 1
	#print "Predicted action: ", np.argmax(predict_y[i,:],axis=0)
	#print "Actual action: ",np.argmax(test_y[i,:], axis = 0)
	#print "-----------------------------------------------------"

print no_match

#Predict
#newstate = GetState()
#model.predict()