import random
import nltk
import lmcorpus
from nltk.corpus import stopwords
from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy
import math
import re
import time
import dateutil.parser

FREQ_NOUN_TOP_N = 10
FREQ_ENTITY_TOP_N = 200
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
REGEX_DATE = '((?:January|February|March|April|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.]{0,1}[\\s][0-3]{0,1}[0-9],{0,1}[\\s][2][0][1][67])|([2][0][1][67]\/[0-9]{0,1}[0-9]\/[0-3]{0,1}[0-9])|((?:January|February|March|April|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\\s][0-3]{0,1}[0-9])'
lm_stop_words = ["moveon.org","thinkprogress", "http", "https", "amazon.com", "math.floor", "constructor", "typeerror", "moveon", "size=medium", "townhall", "breitbart","malkin", "typeerror"]

class DateFinder(object):
	def findLatestDate(self, fileid):
		p = re.compile(REGEX_DATE, re.IGNORECASE)
		text = lmcorpus.raw(fileid)
		dates = p.findall(text)
		ots = 0
		founddate = None
		if(len(dates) == 0):
			return (0,"No date found.")
		for dategroup in dates:
			for datefound in dategroup:
				#print(datefound)
				if(len(datefound) > 0):
					try:
						dt = dateutil.parser.parse(datefound)
						timestamp = time.mktime(dt.timetuple())
					except(ValueError,OverflowError):
						timestamp = 0
					if(ots < timestamp):
						founddate = dt
						ots = timestamp
		return (ots,founddate)
		
class SubjectTrigramTagger(object):
    def __init__(self, train_sents):
        t0 = DefaultTagger('NN')
        t1 = UnigramTagger(train_sents, backoff=t0)
        t2 = BigramTagger(train_sents, backoff=t1)
        self.tagger = TrigramTagger(train_sents, backoff=t2)

    def tag(self, tokens):
        return self.tagger.tag(tokens)

class Binarizer:
	__tokens = {}
	__lastIndex = 0
	__counts = {}
	def showTokens(self):
		return self.__tokens
	
	def showCounts(self):
		return self.__counts
		
	def addTokens(self, tokenSet):
		for token in tokenSet:
			if(token not in self.__tokens):
				self.__tokens[token] = self.__lastIndex
				self.__lastIndex+=1
			if(token in self.__counts):
				self.__counts[token] = self.__counts[token] + 1
			else:
				self.__counts[token] = 1
		return self.__tokens
	
	def pruneBinarySet(self, support):
		self.__tokens = [token for token in self.__tokens if self.__counts[token] >= support]
		
	def getBinarySet(self, tokenSet):
		#for each token in __tokens, if token is in tokenSet, append 1 (otherwise append 0)
		return [(1 if key in tokenSet else 0) for key in self.__tokens]
		
	def importTokens(self, modelFileId, dropFromEnd):
		raw = lmcorpus.raw(modelFileId)
		attrLine = raw.split("\n")[0]
		attrLine = attrLine.replace("\"", "")
		attrLine = attrLine.split(",")
		lattr = len(attrLine)
		attrLine = attrLine[0:lattr - dropFromEnd]
		self.addTokens(attrLine)
		return attrLine
		
class LaughingMonkeyLearner:
	__tagger = None
	__binarizer = None
	__datefinder = None
	def buildTagger(self):
		train_sents = nltk.corpus.brown.tagged_sents()
		train_sents += nltk.corpus.conll2000.tagged_sents()
		train_sents += nltk.corpus.treebank.tagged_sents()
		trigram_tagger = SubjectTrigramTagger(train_sents)
		return trigram_tagger
		
	def extractNamedEntities(self, fileid):
		names = []
		sents = lmcorpus.sents(fileid)[4:-10]
		sentences = [nltk.word_tokenize(sent) for sent in sents]
		sentences = [nltk.pos_tag(sent) for sent in sentences]
		for tagged_sentence in sentences:
			for chunk in nltk.ne_chunk(tagged_sentence):
				if type(chunk) == nltk.tree.Tree:
					names.append(' '.join([c[0].lower() for c in chunk]))
		return names
		
	def getArticleSubjects(self, fileid):
		words = lmcorpus.words(fileid)
		#print(words)
		swords = stopwords.words('english') + lm_stop_words
		stop = set(swords)
		words = [word.lower() for word in words if word not in stop and len(word) > 3 and len(word) < 20]
		fdist = nltk.FreqDist(words)
		tags = nltk.pos_tag(words)
		nouns = [word for (word,tag) in tags if tag == "NN"]
		#print(nouns)
		most_freq_nouns = [w for w, c in fdist.most_common(FREQ_NOUN_TOP_N) if w in nouns]
		#entities = self.extractNamedEntities(fileid)
		#fdist2 = nltk.FreqDist(entities)
		#most_freq_entities = [e for e,c in fdist2.most_common(FREQ_ENTITY_TOP_N)]
		#most_freq_entities = [key for key in fdist2 if fdist2[key] > entitySupport]
		subject_nouns = set([noun for noun in most_freq_nouns])# if noun in most_freq_entities])
		subjects = []
		subjects += [subjn for subjn in subject_nouns]
		#print(subjects)
		return subjects

	def getSVOs(self, fileid, subject):
		if(self.__tagger == None):
			self.__tagger = self.buildTagger()
		sentences = lmcorpus.sents(fileid)
		sentences = [nltk.word_tokenize(sent) for sent in sentences]
		sentences = [sentence for sentence in sentences if subject in
					[word.lower() for word in sentence]]
		tagged = [self.__tagger.tag(sent) for sent in sentences]
		svos = [self.getSVO(sentence, subject) for sentence in tagged]
		return svos

	def swapNaN(self, v, r):
		x = float(v)
		if (math.isnan(v)):
			return r
		return v

	def getArticleSentiment(self, fileid):
		sentences = lmcorpus.sents(fileid)
		sid = SentimentIntensityAnalyzer()
		sentiments = [sid.polarity_scores(sentence) for sentence in sentences]
		ret = []

		mn = numpy.mean([self.swapNaN(s["pos"],0) for s in sentiments if s["neu"] < 0.8])
		ret.append(self.swapNaN(mn, 0))
		
		mn = numpy.mean([self.swapNaN(s["neg"],0) for s in sentiments if s["neu"] < 0.8])
		ret.append(self.swapNaN(mn, 0))
		
		mn = numpy.mean([self.swapNaN(s["neu"],0) for s in sentiments if s["neu"] < 0.8])
		ret.append(self.swapNaN(mn, 0))
		
		return ret

	def getSVO(self, sentence, subject):
		subject_idx = next((i for i, v in enumerate(sentence)
						if v[0].lower() == subject), None)
		data = {'subject': subject}
		for i in range(subject_idx, len(sentence)):
			found_action = False
			for j, (token, tag) in enumerate(sentence[i+1:]):
				if tag in VERBS:
					data['action'] = token
					found_action = True
				if tag in NOUNS and found_action == True:
					data['object'] = token
					data['phrase'] = sentence[i: i+j+2]
					return data
		return {}

	def resetBinarizer(self, binarizer = None):
		self.__binarizer = binarizer
	
	def primeBinarizer(self, fileid):
		if(self.__binarizer == None):
			self.__binarizer = Binarizer()
		subjects = self.getArticleSubjects(fileid)
		self.__binarizer.addTokens(subjects)
		return self.__binarizer
	
	def getArticleTimestamp(self, fileid):
		if(self.__datefinder == None):
			self.__datefinder = DateFinder()
		dateinfo = self.__datefinder.findLatestDate(fileid)
		return [dateinfo[0]]
		
	def getArticleAttributes(self, fileid, support):
		labels = []
		attributes = []
		bin = []
		labels += lmcorpus.getCorpusName(fileid)
		bin += self.getArticleSubjects(fileid)
		print(fileid,bin)
		#print(lmcorpus.raw(fileid))
		#for subject in bin:
		#	print([(svo["subject"],svo["action"],svo["object"]) for svo in self.getSVOs(fileid, subject) if len(svo) > 0])
		try:
			self.__binarizer.pruneBinarySet(support)
			attributes += self.__binarizer.getBinarySet(set(bin))
		except AttributeError:
			print("Couldn't binarize subjects.  Results will only include sentiments.  Did you prime the Binarizer?")
			print()
		attributes += self.getArticleSentiment(fileid)
		attributes += self.getArticleTimestamp(fileid)
		return (attributes, labels, fileid)

	def escape(self, token):
		return '"' + token.replace('"', "'").replace("'", "\'") + '"'
		
	def createCSV(self, attrForArticles):
		#labels
		tokens = self.__binarizer.showTokens()
		attrNames = []
		for token in tokens:
			attrNames.append(self.escape(token))
		attrNames += ["pos","neg","neu","timestamp","corpus"]
		#data
		csv = []
		csv.append(','.join(attrNames))
		for attributes in attrForArticles:			
			csv.append(','.join(map(str,attributes[0])) + "," + attributes[1][0])
		return '\n'.join(csv)
	
	#percentage:  50% = 0.5
	def reservoirSample(self, fileList, percentage):
		sample = []
		i = 0
		#get the number of files in percentage
		n = len(fileList) * percentage
		for file in fileList:
			if(i < n):
				sample.append(file)
				i = i + 1
			else:
				if(random.random() < percentage):
					replace = random.randint(0,len(sample)-1)
					sample[replace] = file
		return sample
	
	def generateTrainingSet(self, fileList, support):
		#prime the binarizer
		for fileName in fileList:
			print(fileName)
			self.primeBinarizer(fileName)
		preprocessed = []
		for fileName in fileList:
			preprocessed.append(self.getArticleAttributes(fileName, support))
		csvdoc = self.createCSV(preprocessed)
		return lmcorpus.addToCorpus("attributes", csvdoc)
	
	def generateTestSet(self, fileList, support):
		preprocessed = []
		for fileName in fileList:
			preprocessed.append(self.getArticleAttributes(fileName, support))
		csvdoc = self.createCSV(preprocessed)
		return lmcorpus.addToCorpus("attributes", csvdoc)
	
	def runExperiment2(self, corpora, support):
		return self.runExperiment(corpora, support, dateutil.parser.parse("1/1/2017"), dateutil.parser.parse("2/28/2017"))
		
	def runExperiment(self, corpora, support, startDate=None, endDate=None, maxFiles=None):
		#get the docs
		corporaFiles = []
		min = 1000
		for corpus in corpora:
			smallList = lmcorpus.fileids(corpus)
			corporaFiles.append(smallList)
			if(len(smallList) < min):
				min = len(smallList)

		fileList = []
		for corpList in corporaFiles:
			fileList += corpList[0:min]
			print(len(fileList))

		if(maxFiles != None):
			fileList = fileList[0:maxFiles]
		
		finalList = []
		df = DateFinder()
		for fileid in fileList:
			dt = df.findLatestDate(fileid)
			if(dt[0] != 0):
				if(dt[1] > startDate and dt[1] < endDate):
					finalList.append(fileid)
				
		fileList = finalList
		trainingFiles = self.reservoirSample(fileList, 0.6)
		testFiles = [file for file in fileList if file not in trainingFiles]
		training = self.generateTrainingSet(trainingFiles, support)
		test = self.generateTestSet(testFiles, support)
		ret = {}
		ret["training"] = training
		ret["test"] = test
		return ret