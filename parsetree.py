import re
from numpy import prod
from binarylr import BinaryLR

topNodePtn = re.compile("^\((\S+) (.+)\)$")

class ParseTree(object):
	def __init__(self, textRepresentation, tagList, classifier, splitProbPredictionThreshold=0.5, parent=None):
		"""
		Constructor for ParseTree class.
		textRepresentation:
			A string describing the parse tree's structure and content. Example from data set:
			"(ROOT (S (NP (PRP He)) (VP (VBD fought) (PP (IN in) (NP (DT the) (NNP Vietnam) (NNP War)))) (. .)))"
		tagList:
			The set of all possible tags (e.g. ROOT, S, NP, PRP, VP, etc.) in the sentence parser's specifications.
		classifier:
			The classifier to be used for predicting content removal.
		parent:
			The parent node. Should be omitted in most cases when manually creating a ParseTree instance.
		"""
		self.splitProbPredictionThreshold = splitProbPredictionThreshold
		self.tagList = tagList
		self.classifier = classifier
		self.parent = parent
		match = topNodePtn.match(textRepresentation)
		if match is None:
			raise ValueError("The provided textRepresentation does not match the required format.")
		self.tag = match.group(1)
		contentStr = match.group(2)
		self.children = []
		if contentStr[0] == "(":
			self.hasChild = True
			parenthSliceEndpoints = [0, 0]
			while True:
				parenthSliceEndpoints = ParseTree.findFirstTopLevelParenthPair(contentStr, parenthSliceEndpoints[1])
				if not parenthSliceEndpoints:
					break
				self.children.append(ParseTree(contentStr[parenthSliceEndpoints[0]:parenthSliceEndpoints[1]], self.tagList, self.classifier, self.splitProbPredictionThreshold, self))
			self.word = ""
		else:
			self.hasChild = False
			self.word = contentStr
		if parent is None:
			self.initFeatures()

	def calcNumDescendants(self):
		"""
		Returns the number of descendants of self.
		"""
		if self.hasChild:
			self.numDescendants = sum(child.calcNumDescendants() + 1 for child in self.children)
		else:
			self.numDescendants = 0
		return self.numDescendants

	def calcMaxDescendantDepth(self):
		"""
		Recursively finds, records, and returns the maximum depth of any descendant of self.
		Assumes that node depths have been calculated.
		"""
		if self.hasChild:
			self.maxDescendantDepth = max([child.calcMaxDescendantDepth() for child in self.children])
		else:
			self.maxDescendantDepth = self.depth
		return self.maxDescendantDepth

	def initFeatures(self):
		"""
		Initializes features used for content-removal prediction.
		"""
		tagArray = [0] * len(self.tagList)
		tagArray[self.tagList.index(self.tag)] = 1
		parentTagArray = [0] * len(self.tagList)
		if self.parent is not None:
			parentTagArray[self.tagList.index(self.parent.tag)] = 1
			numSiblings = len(self.parent.children) - 1
			self.depth = self.parent.depth + 1
		else:
			numSiblings = 0
			self.depth = 0
		numChildren = len(self.children)
		self.calcNumDescendants()
		for child in self.children:
			child.initFeatures()
		self.calcMaxDescendantDepth()
		dropChildProb = 0
		self.featureArray = tagArray + parentTagArray + [
			numSiblings,
			self.depth,
			numChildren,
			self.numDescendants,
			self.maxDescendantDepth,
			dropChildProb
		]

	def getFeatureVector(self):
		"""
		Returns a list of features for content-removal prediction.
		"""
		if(self.hasChild):
			self.featureArray[-1] = 1 - prod([1 - child.predictDropProb() for child in self.children])
		else:
			self.featureArray[-1] = 0 # The above line results in errors in this case. Set to 0 explicitly instead.
		return self.featureArray

	def predictDropProb(self):
		"""
		Returns the classifier's prediction (as a probability in the range [0, 1]) for whether or not some part of the node should be dropped.
		"""
		return self.classifier.classify(self.getFeatureVector(), True)

	def predictSplitProb(self):
		"""
		Returns the classifier's prediction (as a probability in the range [0, 1]) for whether or not the sentence should be split.
		"""
		return self.children[0].predictDropProb() # ROOT.children[0] should be the S node. (This should only be called on ROOT.)

	def predictSplit(self):
		"""
		Returns the classifier's prediction (either 0 or 1) for whether or not the sentence should be split.
		"""
		return True if self.predictSplitProb() > self.splitProbPredictionThreshold else False

	def learn(self, shouldHaveSplit):
		"""
		Calls the classifier's learn() method to teach it whether or not the most recently classified sentence should have been split.
		"""
		self.classifier.learn(1 if shouldHaveSplit else 0)

	def printIndented(self, indentStr=""):
		"""
		Prints an indented textual representation of the parse tree.
		"""
		headIndentStr = indentStr[:-4] + (" +- " if indentStr else "")
		mainIndentStr = indentStr + " |  "
		lastIndentStr = indentStr + "    "
		if self.hasChild:
			print(headIndentStr + self.tag)
			for child in self.children[:-1]:
				child.printIndented(mainIndentStr)
			for child in self.children[-1:]:
				child.printIndented(lastIndentStr)
		else:
			print(headIndentStr + self.tag + " " + self.word)

	def findFirstTopLevelParenthPair(string, startIdx=0):
		"""
		Returns endpoints [open, close] of a slice such that string[open:close] == "(...)"
		"""
		openParenth = -1
		closeParenth = -1
		for i in range(startIdx, len(string)):
			if string[i] == "(":
				openParenth = i
				break
		if openParenth == -1:
			return None
		nestingDepth = 1
		for i in range(openParenth + 1, len(string)):
			if string[i] == ")":
				nestingDepth -= 1
				if nestingDepth == 0:
					closeParenth = i
					break
			elif string[i] == "(":
				nestingDepth += 1
		if closeParenth == -1:
			return None
		return [openParenth, closeParenth + 1]