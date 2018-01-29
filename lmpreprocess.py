import sys
import lmcorpus
import lm
import lmlearning
import hashlib
import base64
import time
import lmnetwork		
import numpy as np

def preprocess(attrFileId):
	print("File Preprocess: ", attrFileId)
	attrs = []
	classes = []
	rows = lmcorpus.raw(attrFileId).split("\n")[1:]
	for sent in rows:
		words = [x.strip() for x in sent.split(',')]
		size = len(words)
		attrs.append(words[0:size - 1])
		classes.append(words[size - 1])
	attributes = np.asarray(attrs[0:size-2])
	labels = np.asarray([[(1 if (c == 'Liberal') else 0)] for c in classes[0:size-2]])
	return (attributes,labels)
	
def train(attrFileId):
	attrs = preprocess(attrFileId)
	attributes = attrs[0]
	labels = attrs[1]
	lmnet = lmnetwork.NeuralNetwork()
	model = lmnet.buildNetwork(attributes, labels)	
	m = lmnet.saveModel(model, attrFileId)	
	return m

def compare(trainingFileId, attrFileId):
	bin = lmlearning.Binarizer()
	bin.importTokens(trainingFileId, 5)
	print(bin.showTokens())
	lml = lmlearning.LaughingMonkeyLearner()
	lml.resetBinarizer(bin)
	testSet = lml.generateTestSet([attrFileId], 0)
	print(bin.showCounts())
	
def evaluate(trainingFileId, attrFileId):
	#pre-process it
	bin = lmlearning.Binarizer()
	bin.importTokens(trainingFileId, 5)
	#print(bin.showCounts())
	lml = lmlearning.LaughingMonkeyLearner()
	lml.resetBinarizer(bin)
	testSet = lml.generateTestSet([attrFileId], 0)
	#find model if it exists
	modelName = trainingFileId + ".model"
	model = None
	if(lmcorpus.exists(modelName)):
		lmn = lmnetwork.NeuralNetwork()
		model = lmn.loadModel(trainingFileId)
	else:
		model = train(trainingFileId)
	attrs = preprocess(testSet)
	attributes = attrs[0]
	#rate it
	prediction = model.predict_classes(attributes)
	#update with score
	return prediction