#!/usr/bin/env python3

import gzip
import pickle
import os
import unicodedata
import numpy
import random
import keras
import string
import io
import collections
import operator

from PIL import Image
from os.path import isfile, isdir, join
from os import listdir, remove
from clean_text import concatenate_files, clean_text

# Returns the number of author found in the directory "Text"
def get_auth_number():
	directory = "train"
	count = 0
	for subdir in next(os.walk(directory))[1]:
		if len(listdir(join(directory, subdir))) > 0:
			count = count+1
	return count

# Returns a sorted list
def init_auth_names(target):
	if target != None and target == []:
		for subdir in next(os.walk(join("train")))[1]:
			if listdir(join(join("train"), subdir)):
				target.append(subdir)
	target.sort()
	return target

def data_to_JSON(prediction, authors, state, idx, max_len):
	data = []
	for i in range(len(authors)):
		data.append({"axis": authors[i], "value": round(float(prediction[i]),4)})
	
	return [[data], {"predicted author":float(numpy.argmax(prediction)), "state":state, "current_index":idx, "max_len":max_len}]

def own_to_categorical(data, nb_author):
	data_new = []
	data_tmp = []
	for element in data:
		data_tmp = [0] * nb_author
		data_tmp[element] = 1
		data_new.append(data_tmp)

	assert len(data_new) == len(data), "[!] Error during categorical transformation: length of returned array different from original array."

	return numpy.array(data_new)

def prepare_text(type):

	print("[*] Concatenation started\n")

	if type.lower() == 'train':
		inpath = "train"

		dirs = (file for file in listdir(inpath) if isdir(join(inpath, file)))

		for directory in dirs:
			print(directory)
			concatenate_files(join(inpath, directory), join(inpath, directory, 'Result', 'input_' + directory.lower() + '_tmp.txt'))
			clean_text(	join(inpath, directory, 'Result', 'input_' + directory.lower() + '_tmp.txt'), 
						join(inpath, directory, 'Result', 'input_' + directory.lower() + '.txt'),
						zip_files=False )
			remove(join(inpath, directory, 'Result', 'input_' + directory.lower() + '_tmp.txt'))

	elif type.lower() == 'test':
		inpath = "test"
		filename = "test_author"

		concatenate_files(inpath, join(inpath, 'Result', filename + '_tmp.txt'))
		clean_text(join(inpath, 'Result', filename + '_tmp.txt'), join(inpath, 'Result', filename + '.txt'), zip_files=False)
		remove(join(inpath, 'Result', filename + '_tmp.txt'))


	print("\n[*] Concatenation ended")

def get_sample_context(text, hyperparameters):

	# Text preprocessing
	text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
	text = text.replace(' ', '')
	text = text.lower()

	translator = str.maketrans('', '', string.punctuation)

	# Initialization
	inputDirectory = 'test/Result'

	file_list = [join(inputDirectory, f) for f in listdir(inputDirectory) if isfile(join(inputDirectory, f)) and '.txt' in f]
	split_value = 30
	context = ''

	# Looking for each file if it contain text. 
	for file in file_list:
		with io.open(file, 'r', encoding='utf-8-sig') as input_file:
			list_lines = input_file.readlines()

		for index, line in enumerate(list_lines):

			# Line processing : removing accents, punctuation, spaces and uppercase
			line = ''.join((c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn'))
			line = line.translate(translator)
			line = line.replace(' ', '')
			line = line.lower()
			
			# We look if the text we've given as an input is in the line. To do that, we will look at chunks of texts -> 30 firts characters, then from the 30th character to the 60th and finally
			# from the 60th to the 90th. There is very low failure probability (line break) thanks to this splitting.
			for i in range(1,4):
				count_char = 0
				already_closed = False
				beg_split = (i-1)*split_value
				end_split = i*split_value

				if text[beg_split:end_split] in line:
					# We take five lines before and ten lines after the spotted line
					for i in range(-5,11):
						# In order to spot the relative sample
						if i == 0:
							context += '[['
						# If we've reached the firts line of the sample, we start counting characters. Therefore, we'll now when to close the bracket	
						elif i > 0:
							count_char += len(list_lines[index + 1])
						
						# We remove the last \n of the line, insert a closing bracket and add a \n again
						if count_char > hyperparameters['max_features'] and not already_closed:
							context = context[:-1]
							context += ']]\n'
							already_closed = True

						context += list_lines[index + i]
					return context

def load_text_from_save():

	with gzip.GzipFile(join('train', 'formatted_data_train.pkl.gzip'), 'rb') as pkl_file:
	  x_train, y_train, x_test, y_test = (pickle.load(pkl_file))

	return [x_train, y_train, x_test, y_test]

def load_test_from_save():

	with gzip.GzipFile(join('test', 'formatted_data_test.pkl.gzip'), 'rb') as pkl_file:
	  data_vector = (pickle.load(pkl_file))

	return data_vector

def load_data_test(hyperparameters):
	# The following loads and format the data stored in the folder named "Test"
	# The architecture must be the following:
	# Text --| test_*.txt
	# If more than one file is submitted, the program will exit

	print('\n[*] Loading test data')

	directory = "test/Result"

	data_vector = []
	train_set_x = []
	train_set_y = []
	test_set_x = []
	test_set_y = []

	letter_vector = [0] * len(hyperparameters['alphabet'])
	example_vector = []

	list_files = listdir(directory)
	assert not all('*.txt' in file for file in list_files), "[!] There must be only one file to test. %d file(s) found." %len(list_files)

	for file in listdir(directory):
		if 'test_' in file:
			i = 1
			with open(join(directory, file), "r") as text:
				example_vector = []
				for line in text:
					line = ''.join((c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn'))
					for character in line.lower():
						if character in hyperparameters['alphabet']:
							letter_vector = [0] * len(hyperparameters['alphabet'])
							letter_vector[hyperparameters['alphabet'].index(character)] = 1
							example_vector.append(letter_vector)
							if (i%hyperparameters['max_features']) == 0:
								data_vector.append(numpy.array(example_vector))
								example_vector = []
							i+=1

	data_vector = keras.preprocessing.sequence.pad_sequences(data_vector)

	with gzip.GzipFile(join('test', 'formatted_data_test.pkl.gzip'), 'wb') as pkl_file:
	  pickle.dump(data_vector, pkl_file)

	return data_vector

def load_data_text(hyperparameters):
	# The following loads and format the data stored in the folder named "Text"
	# The architecture must be the following:
	# Text --| Author1 --| Result --| input_*.txt
	#		| Author2 --| Result --| input_*.txt
	#		| Author3 --| Result --| input_*.txt

	print('\n[*] Loading data')

	directory = "train"

	data_vector = []
	train_set_x = []
	train_set_y = []
	test_set_x = []
	test_set_y = []

	count_author = -1
	letter_vector = [0] * len(hyperparameters['alphabet'])
	example_vector = []

	for subdir in next(os.walk(directory))[1]:
		if listdir(join(directory, subdir)):
			hyperparameters['target_names'].append(subdir)

	hyperparameters['target_names'].sort()

	for subdir in hyperparameters['target_names']:
		count_author += 1
		for file in listdir(join(directory, subdir, 'Result')):
			if 'input_' in file:
				i = 1
				with open(join(directory, subdir, 'Result', file), "r") as text:
					target = count_author
					example_vector = []
					for line in text:
						line = ''.join((c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn'))
						for character in line.lower():
							if character in hyperparameters['alphabet']:
								letter_vector = [0] * len(hyperparameters['alphabet'])
								letter_vector[hyperparameters['alphabet'].index(character)] = 1
								example_vector.append(letter_vector)
								if (i%hyperparameters['max_features']) == 0:
									data_vector.append((numpy.array(example_vector), target))
									example_vector = []
								i+=1

	assert hyperparameters['target_names'] == init_auth_names([]), '[!] Two set of authors found.'
	# assert hyperparameters['target_names'] == fixed_author_names, '(!!!) New target names different from fixed_author_names (!!!)'

	random.shuffle(data_vector)

	for element in data_vector:
	  train_set_x.append(element[0])
	  train_set_y.append(element[1])

	dim = 0.7*len(data_vector)

	test_set_x = train_set_x[int(dim):]
	test_set_y = train_set_y[int(dim):]
	train_set_x = train_set_x[:int(dim)]
	train_set_y = train_set_y[:int(dim)]

	train_set_x = keras.preprocessing.sequence.pad_sequences(train_set_x)
	test_set_x = keras.preprocessing.sequence.pad_sequences(test_set_x)

	train_set_y = keras.utils.np_utils.to_categorical(train_set_y, get_auth_number())
	test_set_y = keras.utils.np_utils.to_categorical(test_set_y, get_auth_number())

	with gzip.GzipFile(join('train', 'formatted_data_train.pkl.gzip'), 'wb') as pkl_file:
	  pickle.dump((train_set_x, train_set_y, test_set_x, test_set_y), pkl_file)


	rval = [train_set_x, train_set_y, test_set_x, test_set_y]

	return rval

# Convert a one-hot encoded sentence to a readable sentence. Same goes for the author if given.
def hot_to_string(hyperparameters, matrix=None, author=None):

	if matrix != None:
		sentence = ''
		for element in matrix:
			if 1 in element:
				sentence += hyperparameters['alphabet'][list(element).index(1)]
			else:
				sentence += ' '

	if author != None:
		author = hyperparameters['target_names'][author]

	return (sentence, author)

# Convert a sample to an image
# The darker the most used
def sample_to_image(sample, ratio=1):
	size = (len(sample[0]), len(sample))
	real_picture = []
	list_sample = []
	for element in sample:
		list_sample.append(tuple(element))
	occurences_count = collections.Counter(list_sample)
	max_occ = max(occurences_count.items(), key=operator.itemgetter(1))[1]

	for element in sample:
		for value in element:
			if value == 0:
				real_picture.append(tuple([0,0,0]))
			else:
				occ = occurences_count[tuple(element)]
				rgb_value = int((occ / max_occ) * 255)
				real_picture.append(tuple([rgb_value, rgb_value, rgb_value]))

	im = Image.new("RGB", size)
	im.putdata(real_picture)
	if ratio != 1:
		im = im.resize((size[0] * ratio, size[1] * ratio))
	im = im.rotate(-90, expand=True)
	im.show()