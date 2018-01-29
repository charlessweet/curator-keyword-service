import urllib.request;
import json
import importlib
import webbrowser
import ast
import os
import uuid
import nltk
import requests
import shutil

import os 
root_path = os.path.dirname(os.path.realpath(__file__))

corpus_root = "corpus"
article_types = ["opinion","news","feature","review","sports","column","press_release","letter_to_editor"]

def createCorpus(tagName):
	subtag_path = corpus_root + "/" + tagName
	if not os.path.exists(subtag_path):
		os.makedirs(subtag_path)
	return subtag_path

def articleFullPath(tag, fileid):
	return root_path + "/" + corpus_root + "/" + tag + "/" + fileid

def articlePath(tag, fileid):
	return corpus_root + "/" + tag + "/" + fileid

#adds the text to the corpus under the associated tag	
def addToCorpus(tag, text, fileid = None):
	createCorpus(tag) #just in case it doesn't exist
	if(fileid == None):
		fileid = str(uuid.uuid4())[:8]
	article_path = corpus_root + "/" + tag + "/" + fileid
	with open(article_path, 'w', encoding='utf-8') as text_file:
		print(text, file=text_file)
	return fileid

def removeFromCorpus(tag, fileid):
	article_path = articlePath(tag, fileid)
	os.remove(article_path)
	return article_path

def moveToCorpus(oldtag, newtag, fileid):
	old_path = articlePath(oldtag, fileid)
	new_path = articlePath(newtag, fileid)
	createCorpus(newtag)
	shutil.copyfile(old_path, new_path);
	removeFromCorpus(oldtag, fileid)
	return new_path
	
def tagArticle(tag, fileid):
	return addToCorpus(tag, fileid)

def untagArticle(tag, fileid):
	return removeFromCorpus(tag, fileid)

def fileids(corpus = None):
	file_list = []
	for path, subdirs, files in os.walk(corpus_root):
		if(corpus == None or corpus in path):
			file_list += files
	return list(set(file_list))

def words():
	files = fileids()
	found_words = []
	found_words += [words(file) for file in files]
	
def words(fileid = None):
	if(fileid == None): #entire corpus
		found_words = []
		files = fileids()
		found_words += [words(file) for file in files]
		return found_words
	#single file
	data = raw(fileid)
	if(data != None):
		return nltk.tokenize.word_tokenize(data)
	return None
	
def sents(fileid = None):
	if(fileid == None): #entire corpus
		found_sents = []
		files = fileids()
		for file in files:
			if(file != "README"):
				found_sents += sents(file)
		return found_sents
	#single file
	data = raw(fileid)
	if(data != None):
		return nltk.tokenize.sent_tokenize(data)
	return None

def raw(fileid):
	for path, subdirs, files in os.walk(corpus_root):
		for f in files:
			if f == fileid:
				file_path = path + "/" + f
				with open(file_path, "r", encoding='utf-8') as text_file:
					return text_file.read()
	return None

def exists(fileid):
	for path, subdirs, files in os.walk(corpus_root):
		for f in files:
			if f == fileid:
				return True
	return False

def getCorpusName(fileid):
	tags = []
	for path, subdirs, files in os.walk(corpus_root):
		for f in files:
			if f == fileid:
				tags.append(os.path.split(path)[1])
	return tags

def filesInCorpora(corpora):
	fileList = []
	for corpus in corpora:
		fileList += fileids(corpus)
	return fileList

def contains(fileid, word):
	wordGrp = words(fileid)
	wordExists = (word in wordGrp)
	print(wordExists, fileid)
	return wordExists
		
def findFileIdContainingWord(corpora, word):
	files = filesInCorpora(corpora)
	retFiles = []
	for file in files:
		rw = raw(file)
		if(word in rw):
			retFiles.append(file)
	return retFiles