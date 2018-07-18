#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import os
import re
import io
import csv

def clean_entry(text):
	brackets = "\[.*?\]"
	backslash = "\s"
	digits = "[0-9]+"

	brackets_p = re.compile(brackets)
	backslash_p = re.compile(backslash)
	digits_p = re.compile(digits)
	
	text = re.sub(brackets_p, '', re.sub(backslash_p, '', text))
	
	return re.sub(digits_p, ';', text).split(';')[1:]

def convert_word2vec():
    with open('word2vec/fr.tsv','r') as tsvin:
        text = clean_entry(tsvin.read())

    words_vec = numpy.load("word2vec/fr.bin.syn0.npy")
    
    word2vec = []

    for i, vect in enumerate(words_vec):
    	word2vec.append((text[i],vect))

    return word2vec

def convert_text_word2vec(max_f):

	directory = "Text"
	alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	count_author = 0
	max_features = max_f

	train_set_x = []
	train_set_y = []
	test_set_x = []
	test_set_y = []



	count_author = -1
	letter_vector = [0] * len(alphabet)
	target_vector = []
	example_vector_train = []
	example_vector_test = []

	target_vector_train = []
	target_vector_test = []

	for subdir in next(os.walk(directory))[1]:
		if os.listdir(os.path.join(directory, subdir)):
			count_author = count_author + 1
		for file in os.listdir(os.path.join(directory, subdir)):
			if '_input' in file:
				i = 1
				with open(os.path.join(directory, subdir, file), "r") as text:
					target = count_author
					example_vector_train = []
					for line in text:
						for word in line:
							print(word)
							for character in word:
								if character in alphabet:
									letter_vector = [0] * len(alphabet)
									letter_vector[alphabet.index(character)] = 1
									example_vector_train.append(letter_vector)
									if (i%max_features) == 0:
										train_set_x.append(numpy.array(example_vector_train))
										train_set_y.append(target)
										example_vector_train = []
									i+=1

			elif '_test' in file:
				i = 1
				with open(os.path.join(directory, subdir, file), "r") as text:
					target = count_author
					example_vector_test = []
					for line in text:
						for word in line:
							for character in word:
								if character in alphabet:
									letter_vector = [0] * len(alphabet)
									letter_vector[alphabet.index(character)] = 1
									example_vector_test.append(letter_vector)
									if (i%max_features) == 0:
										test_set_x.append(numpy.array(example_vector_test))
										test_set_y.append(target)
										example_vector_test = []
									i+=1

if __name__ == '__main__':
    convert_text_word2vec(376)