import sys
import re
import random
import numpy as np
from parsetree import ParseTree
from binarylr import BinaryLR

tagList = [
	'VBG', 'NX', 'NNPS', 'JJS', '#', 'UCP', 'VBZ', 'VBN', 'PRN', "''", 'WHADJP', 'PP', 'LS', 'FRAG', 'NNP', 'JJR',
	':', 'TO', 'WHPP', '``', 'ADVP', 'WHNP', 'VBD', ',', 'PDT', 'WHADVP', '$', 'S', 'RRC', 'RBS', 'CONJP', 'POS',
	'SYM', 'ADJP', '.', 'NP-TMP', 'EX', 'VP', 'NNS', 'VB', 'NN', 'SBAR', 'WDT', '-LRB-', 'DT', 'MD', 'PRT', 'UH',
	'WP', 'WP$', 'SINV', 'WRB', 'X', 'RB', 'PRP', '-RRB-', 'SBARQ', 'SQ', 'PRP$', 'FW', 'CC', 'QP', 'NAC', 'CD',
	'JJ', 'VBP', 'ROOT', 'INTJ', 'NP', 'LST', 'RBR', 'IN', 'RP'
]

defaultLearningRate = 0.1
splitProbPredictionThreshold = 0.9

examplePtn = re.compile("^(?:[^,]*,){4}(?:\"(.*?)\"|([^\"]*?)),(?:\".*?\"|[^\"]*?),((?:Non-)?Split)$")

def getExamples(model, dataFileName):
	"""
	Returns an array of (ParseTree, label) tuples for either training or testing.
	"""
	with open(dataFileName, "r", encoding="utf8") as fp:
		examples = fp.read().split("\n")
	examples = examples[1:]
	for i in range(len(examples)):
		extractedExampleData = examplePtn.match(examples[i]).groups("")
		examples[i] = (ParseTree("".join(extractedExampleData[:2]), tagList, model, splitProbPredictionThreshold), 1 if extractedExampleData[2][0] == "S" else 0)
	return examples

def vote(models, dataFileName, votingThreshold):
	"""
	Tests a voting model made up of multiple classifiers.
	"""
	with open(dataFileName, "r", encoding="utf8") as fp:
		examples = fp.read().split("\n")
	examples = examples[1:]
	truePositives	= 0
	falsePositives	= 0
	trueNegatives	= 0
	falseNegatives	= 0
	for i in range(len(examples)):
		extractedExampleData = examplePtn.match(examples[i]).groups("")
		parseTrees = [ParseTree("".join(extractedExampleData[:2]), tagList, model, splitProbPredictionThreshold) for model in models]
		goldLabel = 1 if extractedExampleData[2][0] == "S" else 0
		if sum([parseTree.predictSplit() for parseTree in parseTrees]) >= votingThreshold:
			if goldLabel:
				truePositives += 1
			else:
				falsePositives += 1
		else:
			if goldLabel:
				falseNegatives += 1
			else:
				trueNegatives += 1
	accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
	precision = truePositives / (truePositives + falsePositives)
	recall = truePositives / (truePositives + falseNegatives)
	f1 = 2/(1/precision + 1/recall)
	print("Accuracy:  {:11.12}%".format(100 * accuracy))
	print("Precision: {:11.12}%".format(100 * precision))
	print("Recall:    {:11.12}%".format(100 * recall))
	print("F1 Score:  {:11.12}%".format(100 * f1))

def bootstrap(models, dataFileName, votingThreshold, numSamples, baseline):
	"""
	Tests a voting model made up of multiple classifiers and compares to a baseline using bootstrap resampling.
	"""
	with open(dataFileName, "r", encoding="utf8") as fp:
		examples = fp.read().split("\n")
	examples = examples[1:]
	truePositives	= 0
	falsePositives	= 0
	trueNegatives	= 0
	falseNegatives	= 0
	for i in range(len(examples)):
		extractedExampleData = examplePtn.match(examples[i]).groups("")
		parseTrees = [ParseTree("".join(extractedExampleData[:2]), tagList, model, splitProbPredictionThreshold) for model in models]
		goldLabel = 1 if extractedExampleData[2][0] == "S" else 0
		if sum([parseTree.predictSplit() for parseTree in parseTrees]) >= votingThreshold:
			if goldLabel:
				examples[i] = [1, 0, 0, 0] # true positive
			else:
				examples[i] = [0, 1, 0, 0] # false positive
		else:
			if goldLabel:
				examples[i] = [0, 0, 0, 1] # false negative
			else:
				examples[i] = [0, 0, 1, 0] # true negative
		# examples[i] = [truePositives, falsePositives, trueNegatives, falseNegatives]
	sampleResults = [None] * numSamples
	for i in range(numSamples):
		[truePositives, falsePositives, trueNegatives, falseNegatives] = np.sum(random.choices(examples, k=len(examples)), axis=0)
		accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
		precision = truePositives / (truePositives + falsePositives)
		recall = truePositives / (truePositives + falseNegatives)
		f1 = 2/(1/precision + 1/recall)
		sampleResults[i] = [accuracy > baseline, precision > baseline, recall > baseline, f1 > baseline]
	[pAccuracy, pPrecision, pRecall, pF1] = np.subtract(1, np.divide(np.sum(sampleResults, axis=0), numSamples))
	print("P-values that statistics fall at or below baseline:")
	print("Accuracy:  {:11.12}".format(pAccuracy))
	print("Precision: {:11.12}".format(pPrecision))
	print("Recall:    {:11.12}".format(pRecall))
	print("F1 Score:  {:11.12}".format(pF1))

def train(model, trainingFileName, numEpochs, learningRate=None):
	"""
	Trains the provided classifier for the given number of epochs.
	"""
	examples = getExamples(model, trainingFileName)
	if learningRate is not None:
		examples[0][0].classifier.learningRate = learningRate
	for i in range(numEpochs):
		print(i)
		random.shuffle(examples)
		for (parseTree, goldLabel) in examples:
			parseTree.predictSplitProb()
			parseTree.learn(goldLabel)

def test(model, testingFileName):
	"""
	Tests the model against the indicated testing-data file.
	"""
	examples = getExamples(model, testingFileName)
	truePositives	= 0
	falsePositives	= 0
	trueNegatives	= 0
	falseNegatives	= 0
	for [parseTree, goldLabel] in examples:
		if parseTree.predictSplit():
			if goldLabel:
				truePositives += 1
			else:
				falsePositives += 1
		else:
			if goldLabel:
				falseNegatives += 1
			else:
				trueNegatives += 1
	accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
	precision = truePositives / (truePositives + falsePositives)
	recall = truePositives / (truePositives + falseNegatives)
	f1 = 2/(1/precision + 1/recall)
	print("Accuracy:  {:11.12}%".format(100 * accuracy))
	print("Precision: {:11.12}%".format(100 * precision))
	print("Recall:    {:11.12}%".format(100 * recall))
	print("F1 Score:  {:11.12}%".format(100 * f1))

def loadAndTrain(trainingFileName, numEpochs, modelOutputFileName=None, learningRate=None, modelInputFileName=None):
	"""
	Loads a classifier from the indicated model file and trains it for the given number of epochs.
	Creates a new model if no model file is provided.
	Saves the model to disk if an output file is indicated.
	Returns the model.
	"""
	numEpochs = int(numEpochs)
	if learningRate is None:
		learningRate = defaultLearningRate;
	else:
		learningRate = float(learningRate)
	if modelInputFileName is not None:
		model = BinaryLR.loadFromFile(modelInputFileName)
	else:
		model = BinaryLR(2 * len(tagList) + 6, learningRate)
	train(model, trainingFileName, numEpochs, learningRate)
	if modelOutputFileName is not None:
		model.saveToFile(modelOutputFileName)
	return model

def loadAndTest(testingFileName, modelInputFileName):
	"""
	Loads a classifier from the indicated model file and tests it against the data from the indicated testing file.
	"""
	test(BinaryLR.loadFromFile(modelInputFileName), testingFileName)

def loadAndVote(testingFileName, votingThreshold, *modelInputFileNames):
	"""
	Loads a classifier from the indicated model file and tests it against the data from the indicated testing file.
	"""
	votingThreshold = int(votingThreshold)
	vote([BinaryLR.loadFromFile(modelInputFileName) for modelInputFileName in modelInputFileNames], testingFileName, votingThreshold)

def loadAndBootstrap(testingFileName, numSamples, baseline, votingThreshold, *modelInputFileNames):
	"""
	Loads a classifier from the indicated model file and tests it against the data from the indicated testing file.
	"""
	numSamples = int(numSamples)
	baseline = float(baseline)
	votingThreshold = int(votingThreshold)
	bootstrap([BinaryLR.loadFromFile(modelInputFileName) for modelInputFileName in modelInputFileNames], testingFileName, votingThreshold, numSamples, baseline)

def showUsage():
	"""
	Show usage statement and exit.
	"""
	print("Usage:")
	print("\tpython s_split.py train trainingFileName numEpochs modelOutputFileName [learningRate [modelInputFileName]]")
	print("\tpython s_split.py test testingFileName modelInputFileName")
	print("\tpython s_split.py vote testingFileName votingThreshold modelInputFileName_0 [...]")
	print("\tpython s_split.py bootstrap testingFileName numSamples baseline votingThreshold modelInputFileName_0 [...]")
	sys.exit(1)

if __name__ == "__main__":
	if len(sys.argv) >= 2:
		if sys.argv[1] == "train":
			if len(sys.argv) >= 5 and len(sys.argv) <= 7:
				loadAndTrain(*sys.argv[2:])
			else:
				showUsage()
		elif sys.argv[1] == "test":
			if len(sys.argv) == 4:
				loadAndTest(*sys.argv[2:])
			else:
				showUsage()
		elif sys.argv[1] == "vote":
			if len(sys.argv) >= 5:
				loadAndVote(*sys.argv[2:])
			else:
				showUsage()
		elif sys.argv[1] == "bootstrap":
			if len(sys.argv) >= 7:
				loadAndBootstrap(*sys.argv[2:])
			else:
				showUsage()
		else:
			showUsage()
	else:
		showUsage()