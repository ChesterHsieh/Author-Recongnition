#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import sys
import os
import argparse
import string
import unicodedata
from os import listdir, remove
from os.path import isfile, join

def getSampleContext(text):

	text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
	text = text.replace(' ', '')
	text = text.lower()
	translator = str.maketrans('', '', string.punctuation)

	inputDirectory = 'gutenberg'

	file_list = [join(inputDirectory, f) for f in listdir(inputDirectory) if isfile(join(inputDirectory, f))]

	context = ''

	for file in file_list:
		with io.open(file, 'r', encoding='utf-8-sig') as input_file:
			list_lines = input_file.readlines()
		for index, line in enumerate(list_lines):
			line = ''.join((c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn'))
			line = line.translate(translator)
			line = line.replace(' ', '')
			line = line.lower()
			if text[:30] in line.lower():
				print(line)
				print(text)
				for i in range(-10,11):
					context += list_lines[index + i]
				return context

def getNameText(text, cmd):
	
	inputDirectory = 'gutenberg'

	file_list = [join(inputDirectory, f) for f in listdir(inputDirectory) if isfile(join(inputDirectory, f))]
	flag_found = False
	nb_file = 0
	cmd += ' '
	for file in file_list:
			with io.open(file, 'r', encoding='utf-8-sig') as input_file:
				for line in input_file:
					line = ''.join((c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn'))
					if text in line:
						flag_found = True

				if flag_found:
					nb_file+=1		
					print(file)
					os.system(cmd + file)
					flag_found = False

	print("Number of file(s) found : " + str(nb_file))

def findAllIncorrectTexts(inputDirectory):

	opening_chars = [u'(', u'[', u'{']
	closure_chars = [u')', u']', u'}']

	flag_chars = [0] * len(opening_chars)

	file_list = [join(inputDirectory, f) for f in listdir(inputDirectory) if isfile(join(inputDirectory, f))]

	incorrectTexts = []

	for file in file_list:
		with io.open(file, 'r', encoding='utf-8-sig', errors='ignore') as input_file:
			flag_chars = [0] * len(opening_chars)
			for line in input_file:
				for char in line:
					if char in opening_chars:
						flag_chars[opening_chars.index(char)] += 1
					elif char in closure_chars:
						flag_chars[closure_chars.index(char)] -= 1
		if flag_chars != [0] * len(opening_chars):
			incorrectTexts.append(file)

	return incorrectTexts

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-s', type=str,default='',help='String to be found in the texts available in the directory "gutenberg".')
	parser.add_argument('--editor', type=str, default='gedit', help='Command launching the text editor, e.g. "gedit" for Gedit or "subl" for Sublime Text.')	
	args = parser.parse_args()

	print(getSampleContext(args.s))