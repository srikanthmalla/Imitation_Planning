import tensorflow as tf
import numpy as np


"""
Description :
We have a dataset of of Joint angles accrding to time stamps
This Feed Forward Neural Network is created to predict the 
next set of joint angles given joint angles at current time step.
"""
#Declare Global variables
LEARNING_RATE = 0.1
HIDDEN_LAYER_1 = 20
HIDDEN_LAYER_2 = 20
NO_EPOCHS = 100

def getDataSet(filename):
	"""
	Gets Data set from given file
	paramters:
	filename = Preferably .csv files

	Return:
	Beautifully shaped Input and Output Data
	Input Data : Joint Angles at time step t
	Output Data : Joint Angles at time step t + 1
	"""
	#Get data set
	data = np.loadtxt(filename,delimiter = ",")
	print data.shape

	#Print for sanity check
	input_data = data[0:-1,1:9]
	print input_data.shape
	output_data = data[1:,1:9]
	print output_data.shape

	return input_data,output_data

def createTestDataSet(filename):
	#Get data set
	data = np.loadtxt(filename,delimiter = ",")
	print data.shape

	#Print for sanity check
	input_data = data[0:-1,1:9]
	print input_data.shape
	output_data = data[1:,1:9]
	print output_data.shape

	#Concatenate input and output
	combined_data = np.concatenate((input_data, output_data), axis = 1)

	#Shuffle Data
	np.random.shuffle(combined_data)

	#Seprate the input and output
	i = combined_data[:,0:8]
	
	o = combined_data[:,8:]

	# return input_data,output_data
	return i, o


def createNetwork(x, inputs, outputs):

	"""
	Description:
	Create a neural Network or 3 layers, Last is output layer

	Paramters:
	input and output data

	Returns:
	Complete COmputation graph of the Neural Network
	Output layer gives continous values of joint angles at time step t + 1
	given joint angles at time step t.
	"""

	#Get Dimensions
	input_dim = inputs.shape[1]
	output_dim = outputs.shape[1]

	#Weights
	weights = {
	'hidden1' : tf.Variable(tf.random_normal([input_dim, HIDDEN_LAYER_1])),
	'hidden2' : tf.Variable(tf.random_normal([HIDDEN_LAYER_1, HIDDEN_LAYER_2])),
	'out' : tf.Variable(tf.random_normal([HIDDEN_LAYER_2, output_dim]))
	}

	#Biases
	biases = {
	'bias1' : tf.Variable(tf.random_normal([HIDDEN_LAYER_1])),
	'bias2' : tf.Variable(tf.random_normal([HIDDEN_LAYER_2])),
	'bias3' : tf.Variable(tf.random_normal([output_dim]))
	}

	#Create COmputational Graph
	layer_1 = tf.add(tf.matmul(x, weights['hidden1']), biases['bias1'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, weights['hidden2']), biases['bias2'])
	layer_2 = tf.nn.relu(layer_2)

	output_layer = tf.matmul(layer_2, weights['out']) + biases['bias3']
	
	return output_layer


def main():
	"""
	Main Implemenation
	"""
	
	#Get Traning Data
	inputs, outputs = getDataSet('trail_1.csv')

	#get Test Data
	inputs_test, outputs_test = createTestDataSet('trail_2.csv')

	#Get Dimensions of Data
	input_dim = inputs.shape[1]
	output_dim = outputs.shape[1]

	#Declare Placeholders
	x = tf.placeholder("float",[None, input_dim])
	y = tf.placeholder("float", [None, output_dim])

	#Construct Model
	prediction = createNetwork(x, inputs, outputs)

	#Define Loss
	loss = tf.reduce_mean(tf.square(prediction - outputs))

	#Define Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

	#Initliaze all Variables
	init = tf.initialize_all_variables()

	with tf.Session() as sess:

		sess.run(init)

		#For Epochs run optimizer
		for i in range(NO_EPOCHS):

			_, cost = sess.run([optimizer, loss], feed_dict = {x : inputs,
																y : outputs})

			print "For Epoch %d, Loss = %f"%(i, cost)

		print "Shits DOne!!"

		#Test the model on test data

		test_cost = tf.reduce_mean(tf.square(prediction - outputs_test))

		print "Loss during testing : ", test_cost.eval({x : inputs_test, 
														y : outputs_test})




if __name__ == "__main__":
	main()