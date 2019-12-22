#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import requests
import re
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from datetime import datetime



def read_data():
	data = pd.read_csv('https://raw.githubusercontent.com/avhadutgadhave/MLOP-s-POC-AWS/master/iris.csv',
	names=[	'sepal_length','sepal_width','petal_length','petal_width','species'])
	data.head()
	return data



def plot(data):
	plotfile=seaborn.pairplot(data, hue="species", size=2, diag_kind="kde")
	plotfile.savefig('static/mat.png')



def preprocessing(data):
	species_lb = LabelBinarizer()
	Y = species_lb.fit_transform(data.species.values)
	FEATURES = data.columns[0:4]
	X_data = data[FEATURES].as_matrix()
	X_data = normalize(X_data)
	return X_data,Y


def train_test(X_data,Y):
	X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3, random_state=1)
	X_train.shape
	return X_train,X_test,y_train,y_test

def training_model(X_train,y_train):
	# Parameters
	learning_rate = 0.01
	training_epochs = 15
	# Neural Network Parameters
	n_hidden_1 = 256 # 1st layer number of neurons
	n_hidden_2 = 128 # 1st layer number of neurons
	n_input = X_train.shape[1] # input shape (105, 4)
	n_classes = y_train.shape[1] # classes to predict
	# Inputs
	X = tf.placeholder("float", shape=[None, n_input])
	y = tf.placeholder("float", shape=[None, n_classes])
	# Dictionary of Weights and Biases
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}
	# Model Forward Propagation step
	def forward_propagation(x):
		# Hidden layer1
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.relu(layer_1)
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.relu(layer_2)
		# Output fully connected layer
		out_layer = tf.matmul(layer_2, weights['out']) + biases['out'] 
		return out_layer
	# Model Outputs
	yhat = forward_propagation(X)
	ypredict = tf.argmax(yhat, axis=1)
	# Backward propagation
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(cost)
	return ypredict,training_epochs,train_op,X,y


def classifier(data):
	# preprocess the data
	species={'setosa':0,'versicolor':1,'virginica':2}
	data.species=[species[item] for item in data.species]
	df = pd.DataFrame(data)
	fun=df[['sepal_length','sepal_width','petal_length','petal_width']]
	cls=df[['species']]
	X1_train, X1_test,y1_train, y1_test  = train_test_split(fun,cls, test_size=0.3, random_state=1)
	X1_train=X1_train.reset_index(drop=True)
	X1_test=X1_test.reset_index(drop=True)
	y1_train=y1_train.reset_index(drop=True)
	y1_test=y1_test.reset_index(drop=True)
	# Specify that all features have real-value data
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
	# Build 3 layer DNN with 10, 20, 10 units respectively.
	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
 	                                       hidden_units=[10, 20, 10],
 	                                       n_classes=3,
 	                                       model_dir="iris_model")
	return classifier


def compute(sepal_len,sepal_wid,petal_len,petal_wid,classifier):
	new_samples = np.array([[sepal_len,sepal_wid,petal_len,petal_wid]], 	dtype=float)
	val=(list(classifier.predict(new_samples)))
	for elm in val:
		if elm==0:
			return 'setosa'
		elif elm==1:
			return 'verginica'
		elif elm==2:
			return 'virsicolor'


def main():
	data=read_data()
	plot(data)
	X_data,Y=preprocessing(data)
	X_train,X_test,y_train,y_test=train_test(X_data,Y)
	ypredict,training_epochs,train_op,X,y=training_model(X_train,y_train)
#	accuracy(X_train,y_train,ypredict,training_epochs,train_op,X,y)
	def accuracy(X_train,y_train,ypredict,training_epochs,train_op,X,y):
        	# Initializing the variables
		init = tf.global_variables_initializer()
		startTime = datetime.now()
		with tf.Session() as sess:
		        sess.run(init)
        	#writer.add_graph(sess.graph)
        	#EPOCHS
		for epoch in range(training_epochs):
        	#Stochasting Gradient Descent
		        for i in range(len(X_train)):
		                summary = sess.run(train_op, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})

		train_accuracy = np.mean(np.argmax(y_train, axis=1) == sess.run(ypredict, feed_dict={X: X_train, y: y_train}))
		test_accuracy  = np.mean(np.argmax(y_test, axis=1) == sess.run(ypredict, feed_dict={X: X_test, y: y_test}))

		print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
        	#print("Epoch = %d, train accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy))
		sess.close()
		print("Time taken:", datetime.now() - startTime)


	classifier1=classifier(data)
	val=compute(1,3,2,1,classifier1)
	print(val)
if __name__=='__main__':
	main()
