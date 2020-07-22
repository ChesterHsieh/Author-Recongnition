#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import random
import numpy
import argparse

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def main():
	parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--sliding_window', dest='sliding_window', action='store_true',
						help='Enables the use of a sliding window. Must provide a size afterwards')
	parser.add_argument('--no_sliding_window', dest='sliding_window', action='store_false',
						help='Disables the use of a sliding window.')
	parser.set_defaults(sliding_window=False)
	parser.add_argument('--sliding_window_size', type=int, default=0,
                    	help='Size of the sliding window used.')
	args = parser.parse_args()

	if args.sliding_window == True and args.sliding_window_size == 0:
		raise argparse.ArgumentError("Can't use sliding without precising its size")

	dataset = load_data(args)

	X = dataset[0]
	y = dataset[1]

	clf = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd',
						learning_rate='adaptive', max_iter=200, shuffle=True,
						verbose=True)

	print('\n[TRAINING STARTING]')

	clf.fit(X, y)

def get_auth_number():
    directory = "Text"
    count = 0
    for subdir in next(os.walk(directory))[1]:
        if len(os.listdir(os.path.join(directory, subdir))) > 0:
            count = count+1
    return count

def load_data(args):
    print('\n[DATA LOADING]')

    # Load and separate the dataset in 3 different vectors : test_set, train_set and valid_set

    slidingWindow = args.sliding_window
    slidingWindowSize = args.sliding_window_size

    directory = "Text"

    train_set = []
    test_set = []
    validation_set = []
    test_files = []
    validation_files = []

    print("\n    [Authors]")

    # for subdir in next(os.walk(directory))[1]:
    #     if len(os.listdir(os.path.join(directory, subdir))) < 3:
    #         print("       [Warning]             Can't load author %s : not enough files available" %(subdir))
        # else:
        #     test_file = random.choice(os.listdir(os.path.join(directory, subdir)))
            # validation_file = random.choice(os.listdir(os.path.join(directory, subdir)))
            # while test_file == validation_file:
            #     validation_file = random.choice(os.listdir(os.path.join(directory, subdir)))
            # test_files.append(test_file)
            # validation_files.append(validation_file)
            # validation_files = ["020.txt", "054.txt", "033.txt", "229.txt", "226.txt", "248.txt", "108.txt", "139.txt", "127.txt"]
            # test_files = ["004.txt", "039.txt", "048.txt", "230.txt", "249.txt", "203.txt", "101.txt", "140.txt", "144.txt"]

    print("       [Number of authors]   %d" %get_auth_number())

    count_author = -1
    letter_vector = [0] * len(alphabet)
    target_vector = []
    example_vector_train = []
    example_vector_valid = []
    example_vector_test = []

    target_vector_train = []
    target_vector_valid = []
    target_vector_test = []

    print("\n    [Texts]")

    if not slidingWindow:

        print("       [Input Type]          Vectors of letters\n")

        for subdir in next(os.walk(directory))[1]:
            if os.listdir(os.path.join(directory, subdir)):
                count_author = count_author + 1
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith(".txt"):
                    if file not in test_files:
                        with open(os.path.join(directory, subdir, file), "r") as text:
                            target = count_author
                            for line in text:
                                for word in line:
                                    for character in word:
                                        if character in alphabet:
                                            letter_vector = [0] * len(alphabet)
                                            letter_vector[alphabet.index(character)] = 1
                                            example_vector_train.append(letter_vector)
                                            target_vector_train.append(target)

                    elif file in test_files:
                        with open(os.path.join(directory, subdir, file), "r") as text:
                            target = count_author
                            for line in text:
                                for word in line:
                                    for character in word:
                                        if character in alphabet:
                                            letter_vector = [0] * len(alphabet)
                                            letter_vector[alphabet.index(character)] = 1
                                            example_vector_test.append(letter_vector)
                                            target_vector_test.append(target)

                    elif file in validation_files:
                        with open(os.path.join(directory, subdir, file), "r") as text:
                            target = count_author
                            for line in text:
                                for word in line:
                                    for character in word:
                                        if character in alphabet:
                                            letter_vector = [0] * len(alphabet)
                                            letter_vector[alphabet.index(character)] = 1
                                            example_vector_valid.append(letter_vector)
                                            target_vector_valid.append(target)

    else:

        print("       [Input Type]          Vectors of letters, based on sliding window")

        memory = []
        letter_nb = 0

        print("       [Input type]          Size of the window: %d" %slidingWindowSize)
        print("       [Input type]          Step: 1")

        for subdir in next(os.walk(directory))[1]:
            if os.listdir(os.path.join(directory, subdir)):
                count_author = count_author + 1
            for file in os.listdir(os.path.join(directory, subdir)):
                if file.endswith(".txt"):
                    if file not in test_files:
                        with open(os.path.join(directory, subdir, file), "r") as text:
                            target = count_author
                            for line in text:
                                for word in line:
                                    for character in word:
                                        if character in alphabet:
                                            if len(memory) < slidingWindowSize:
                                                memory.append(character)
                                            else:
                                                for read_char in memory:
                                                    letter_vector[alphabet.index(read_char)] = 1
                                                example_vector_train.append(letter_vector)
                                                target_vector_train.append(target)
                                                letter_vector = [0] * len(alphabet)
                                                memory = memory[1:]
                                                memory.append(character)                         
                                            
                    elif file in test_files:
                        with open(os.path.join(directory, subdir, file), "r") as text:
                            target = count_author
                            for line in text:
                                for word in line:
                                    for character in word:
                                        if character in alphabet:
                                            if len(memory) < slidingWindowSize:
                                                memory.append(character)
                                            else:
                                                for read_char in memory:
                                                    letter_vector[alphabet.index(read_char)] = 1
                                                example_vector_test.append(letter_vector)
                                                target_vector_test.append(target)
                                                letter_vector = [0] * len(alphabet)
                                                memory = memory[1:]
                                                memory.append(character)
                    
                    elif file in validation_files:
                        with open(os.path.join(directory, subdir, file), "r") as text:
                            target = count_author
                            for line in text:
                                for word in line:
                                    for character in word:
                                        if character in alphabet:
                                            if len(memory) < slidingWindowSize:
                                                memory.append(character)
                                            else:
                                                for read_char in memory:
                                                    letter_vector[alphabet.index(read_char)] = 1
                                                example_vector_valid.append(letter_vector)
                                                target_vector_valid.append(target)
                                                letter_vector = [0] * len(alphabet)
                                                memory = memory[1:]
                                                memory.append(character)

    train_set_x = numpy.array(example_vector_train)
    train_set_y = numpy.array(target_vector_train)
    validation_set_x = numpy.array(example_vector_valid)
    validation_set_y = numpy.array(target_vector_valid)
    test_set_x = numpy.array(example_vector_test)
    test_set_y = numpy.array(target_vector_test)

    # rval = [(train_set_x, train_set_y),
    #         (test_set_x, test_set_y),
    #         (validation_set_x, validation_set_y)]
    
    rval = [train_set_x, train_set_y]

    return rval

if __name__ == '__main__':
	main()