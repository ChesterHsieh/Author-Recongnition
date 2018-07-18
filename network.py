#!/usr/bin/env python3

from utils import get_auth_number, init_auth_names, data_to_JSON, hot_to_string, get_sample_context

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Layer
from keras.layers.normalization import BatchNormalization
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler, Callback
from keras.utils import plot_model, np_utils
from keras.initializers import RandomUniform, RandomNormal

from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from random import shuffle
from shutil import copy as copy_file
from copy import copy, deepcopy
from os import listdir
from os.path import isfile, join
from collections import Counter
from flask import Flask, render_template, request
from multiprocessing import Process

import keras.backend as K
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import unicodedata
import itertools
import gzip
import pickle
import time
import argparse
import signal, psutil
import os
import io
import string
import sys
import numpy
import h5py
import unicodedata
import json
import logging

# -------------------------------------------- HYPERPARAMETERS--------------------------------------------

def print_statistics(args, model, data, save_dir, hyperparameters, author='Unknown'):

	len_data = len(data)

	def predict_best_worst_batch(matrices):

		# In order: sentence, author_predicted, accuracy
		best = ['', '', 0]
		worst = ['', '', 1]
		best_batches = []
		worst_batches = []
		for i in range(len(hyperparameters['target_names'])):
			best_batches.append(deepcopy(best))
			worst_batches.append(deepcopy(worst))

		y_pred = []

		# For each matrix of the input we'll predict its class and determine if it's the best / worst sample
		# we've dealt with so far
		for idx, element in enumerate(matrices):

			matrix_to_send = []

			matrix_to_send.append(element)

			matrix_to_send = numpy.array(matrix_to_send)

			b_pred = model.predict_classes(matrix_to_send, verbose=0)[0]
			b_proba_pred = model.predict_proba(matrix_to_send, verbose=0)[0]
			sys.stdout.write("\r" + str(idx) + ' / ' + str(len(matrices)))
			sys.stdout.flush()
			y_pred.append(b_pred)

			data = data_to_JSON(b_proba_pred, hyperparameters['target_names'], "running", idx, len_data)

			with open('./static/resources/json/probabilities.json', 'w') as outfile:
				json.dump(data, outfile)

			for index_auth, author in enumerate(hyperparameters['target_names']):

				if b_proba_pred[index_auth] > best_batches[index_auth][2]:
					best_batches[index_auth][0], best_batches[index_auth][1] = hot_to_string(hyperparameters, element, index_auth)
					best_batches[index_auth][2] = b_proba_pred[index_auth]

				if b_proba_pred[index_auth] < worst_batches[index_auth][2]:
					worst_batches[index_auth][0], worst_batches[index_auth][1] = hot_to_string(hyperparameters, element, index_auth)
					worst_batches[index_auth][2] = b_proba_pred[index_auth]

		sys.stdout.write("\n")
		return (best_batches, worst_batches, y_pred)

	# Initialization of the names of the authors
	hyperparameters['target_names'] = init_auth_names(hyperparameters['target_names'])

	best, worst, y_pred = predict_best_worst_batch(data)
	y_pred = numpy.array(y_pred)
	print(y_pred)

	with open(author + "BestSampleAnalysis.txt", "a") as analysisFile:
		analysisFile.write("------------------------------------------------")
		analysisFile.write("\n\nTrue author : " + author)
		analysisFile.write("\n\nHighest probability found per author:\n")
		for element in best:
			element.append(get_sample_context(element[0], hyperparameters))
			analysisFile.write("\n[" + element[1] + "] -> {:.5f}".format(element[2]))
			analysisFile.write("\n\nSample:\n")
			analysisFile.write(element[0])
			analysisFile.write("\n\nActual text:\n")
			if element[3] != None:
				analysisFile.write(element[3])
			else:
				analysisFile.write("None")

	with open(author + "WorstSampleAnalysis.txt", "a") as analysisFile:
		analysisFile.write("------------------------------------------------")
		analysisFile.write("\n\nTrue author : " + author)
		analysisFile.write("\n\nLowest probability found per author:\n")
		for element in worst:
			element.append(get_sample_context(element[0], hyperparameters))
			analysisFile.write("\n[" + element[1] + "] -> {:.5f}".format(element[2]))
			analysisFile.write("\n\nSample:\n")
			analysisFile.write(element[0])
			analysisFile.write("\n\nActual text:\n")
			if element[3] != None:
				analysisFile.write(element[3])
			else:
				analysisFile.write("None")

	stats_pred = Counter(y_pred)
	global_probabilities = []

	for i in range(len(hyperparameters['target_names'])):
		global_probabilities.append(stats_pred[i] / len(y_pred))

	print("\nThe predicted author is " + str(hyperparameters['target_names'][global_probabilities.index(max(global_probabilities))]) 
		+ " with a probability of " + str(round(max(global_probabilities) * 100, 2)) + "%.")

	with open('./static/resources/json/probabilities.json', 'w') as outfile:
		json.dump(data_to_JSON(global_probabilities, hyperparameters['target_names'], "stop", len_data, len_data), outfile)

def print_confusion_matrix(args, model, X_test, Y_test, save_dir, hyperparameters):

	# Plot the confusion matrix in a neat way ; can be saved as a PNG file afterwards
	def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues, precision=2):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = numpy.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
			numpy.set_printoptions(precision=precision)
			cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
			for idx_vec, vector in enumerate(cm):
				for idx_element, element in enumerate(vector):
					cm[idx_vec][idx_element] = round(element, 2)
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(cm)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	# Initialization of the names of the authors
	hyperparameters['target_names'] = init_auth_names(hyperparameters['target_names'])

	y_pred = model.predict_classes(X_test)
	
	print("\n\n	 [Predictions]")
	print(y_pred)
	print("\n\n	 [Expected]")
	print(Y_test)

	print("\n\n	 [Reports]")
	print("			 [Classification Report]\n")
	print(classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=hyperparameters['target_names']))
	print("			 [Normalized confusion matrix]\n")
	conf_mat = confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred)
	plt.figure()
	plot_confusion_matrix(conf_mat, classes = hyperparameters['target_names'], normalize = True, title='Normalized confusion matrix', precision=3)
	plt.show()


	if args.no_reports_saved == False:

		filename = 'report_' + time.strftime("%d_%m_%Y_") + time.strftime("%H:%M:%S") + '.rep'

		if args.train == False:
			save_dir = args.save_dir

		orig_stdout = sys.stdout
		with open(join(save_dir, filename), 'w') as output:
			sys.stdout = output
			print("Training : " + time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + '\n\n')
			print(args.m)
			print("[ARCHITECTURE]\n\n")
			print(model.summary(), '\n\n')
			print('\n\n[CLASSIFICATION REPORT]\n\n',classification_report(numpy.argmax(Y_test,axis=1), y_pred,target_names=hyperparameters['target_names']))
			print('\n\n[CONFUSION MATRIX]\n\n', confusion_matrix(numpy.argmax(Y_test,axis=1), y_pred))
			print('\n\n[HYPERPARAMETERS]', '\n\nInitial learning rate : ' + str(hyperparameters['initial_lr']) + '\nLast learning rate : ' + str(K.get_value(model.optimizer.lr)), '\nDecay of ' + str(hyperparameters['decay_rate']) + ' every ' + str(hyperparameters['epoch_decay']) + ' epochs',
				  '\nMax_feature : ' + str(hyperparameters['max_features']), '\nVector size : ' + str(hyperparameters['vect_size']), '\nAlphabet : ' + str(hyperparameters['alphabet']), '\nHidden dim : ' + str(hyperparameters['hidden_dims']),
				  '\nEpochs : ' + str(hyperparameters['epochs']), '\nFilters : ' + str(hyperparameters['filters']), '\nRandom type : RandomUniform')
		sys.stdout = orig_stdout

def train_model(args, x_train, y_train, x_test, y_test, hyperparameters, log_dir):


	def signal_handler(signal, frame):
		model.stop_training = True


	signal.signal(signal.SIGINT, signal_handler)

	print('\n[*] Building Model')
	model = Sequential()

	random_uni = RandomNormal(mean = 0.0, stddev = 0.05, seed = None)

	def custom_sigmoid_activation(x):
		return 1.7159*K.tanh(2/3*x)

	# We initially follow the architecture given here : https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf
	model.add(Conv1D(hyperparameters['filters'],
					 hyperparameters['kernel_size'][0],
					 kernel_initializer=random_uni,
					 input_shape=(hyperparameters['max_features'],hyperparameters['vect_size'])))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	model.add(MaxPooling1D(pool_size = 2, strides=None))

	model.add(Conv1D(hyperparameters['filters'],
					 hyperparameters['kernel_size'][1],
					 kernel_initializer=random_uni,
					 strides=1))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(MaxPooling1D(pool_size = 2, strides=None))

	model.add(Conv1D(hyperparameters['filters'],
					 hyperparameters['kernel_size'][1],
					 kernel_initializer=random_uni,
					 strides=1))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv1D(hyperparameters['filters'],
					 hyperparameters['kernel_size'][1],
					 kernel_initializer=random_uni,
					 strides=1))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv1D(hyperparameters['filters'],
					 hyperparameters['kernel_size'][1],
					 kernel_initializer=random_uni,
					 strides=1))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(MaxPooling1D(pool_size = 2, strides=None))

	model.add(Flatten())

	model.add(Dense(hyperparameters['hidden_dims'], kernel_initializer=random_uni))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(hyperparameters['hidden_dims'], kernel_initializer=random_uni))
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(get_auth_number(), kernel_initializer=random_uni, activation='softmax'))

	sgd = SGD(lr=hyperparameters['initial_lr'], momentum=0.9)

	last_val_loss = float("inf")

	class BoldScheduler(Callback):
		def __init__(self):
			self.last_val_loss = float("inf")

		def on_epoch_end(self, epoch, logs={}):
				
			curr_val_loss = logs.get('val_loss')
			lr = K.get_value(model.optimizer.lr)

			if(self.last_val_loss > curr_val_loss):
				K.set_value(model.optimizer.lr, lr*1.1)
				print("[*] lr changed from {:.6f} to {:.6f}".format(lr, K.get_value(model.optimizer.lr)))
			elif curr_val_loss - self.last_val_loss > 0.001:
				K.set_value(model.optimizer.lr, lr*0.7)
				print("[*] lr changed from {:.6f} to {:.6f}".format(lr, K.get_value(model.optimizer.lr)))

			self.last_val_loss = curr_val_loss
			return

	def scheduler(epoch):
		
		if epoch%epoch_decay == 0 and epoch != 0:
			lr = K.get_value(model.optimizer.lr)
			K.set_value(model.optimizer.lr, lr*decay_rate)
			print("[*] lr changed to {}".format(lr*decay_rate)) 
		return K.get_value(model.optimizer.lr)

	checkpointer = ModelCheckpoint(filepath=log_dir + '/model.hdf5', verbose=1, save_best_only=True, monitor='val_categorical_accuracy')
	tensorVizualisation = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
	earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto')
	lr_decay = LearningRateScheduler(scheduler)
	bold_decay = BoldScheduler()

	model.compile(loss='categorical_crossentropy',
				  optimizer=sgd,
				  metrics=['categorical_accuracy'])

	plot_model(model, to_file='model.png')

	model.fit(x_train, y_train,
			  batch_size=hyperparameters['batch_size'],
			  epochs=hyperparameters['epochs'],
			  verbose=1,
			  validation_data = (x_test, y_test),
			  shuffle=True,
			  callbacks=[checkpointer, tensorVizualisation])


	copy_file(log_dir + '/model.hdf5', args.save_dir + '/last.hdf5')

	return model