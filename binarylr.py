import copy
import numpy as np
from math import exp
from ast import literal_eval

class NotReadyException(Exception):
	"""
	Ths object is not in the right state for the requested operation.
	"""
	pass # This isn't just a temporary placeholder; nothing needs to be here.

class BinaryLR(object):
	"""
	An implementation of the logistic-regression algorithm for binary classification.
	"""
	activationFunctions = {
		"logistic": {
			"funct": lambda z: 1/(1 + exp(-z)),
			"deriv": lambda z, a: max(a * (1 - a), 1e-2)
		}
	}
	activationBounds = [0, 1] # [min, max]
	activationThreshold = 0.5 # activation(potential=0)
	costFunctions = {
		"log-loss": {
			# "funct": lambda a, y: y * log(a) + (1 - y) * log(1 - a),
			"deriv": lambda a, y: -( # use limits in edge cases instead of dividing by 0
				0 if a == y
				else 2e2 * (y - 0.5) if a == (not y) # use large finite value instead of float("inf")
				else np.clip(y/a - (1 - y)/(1 - a), -1e2, 1e2)
			)
		},
		"MSE": {
			# "funct": lambda a, y: 1/2 * (y - a)**2,
			"deriv": lambda a, y: -(y - a)
		}
	}
	MIN_POTENTIAL = -500
	MAX_POTENTIAL = +500
	def __init__(
			self,
			inputSources,
			learningRate,
			activation=activationFunctions["logistic"]["funct"],
			dadz=activationFunctions["logistic"]["deriv"],
			# cost=BinaryLR.costFunctions["log-loss"]["funct"],
			dCda=costFunctions["log-loss"]["deriv"],
			activationBounds=activationBounds,
			activationThreshold=activationThreshold
		):
		"""
		Constructor for BinaryLR class.
		inputSources: int or list; TODO: further documentation
		learningRate: TODO: documentation
		activation: activation function of dot product (logistic function by default)
		dadz: derivative of activation function
		dCda: partial derivative of cost function w.r.t. activation
		"""
		self.numOutputs = 0
		self.outputWeightedErrors = []
		if type(inputSources) is int:
			self.numInputs = inputSources
			self.inputSources = None
			self.inputStates = None
		else:
			self.numInputs = len(inputSources)
			self.inputSources = inputSources
			for inputSource in inputSources:
				inputSource.numOutputs += 1
			self.inputSources += [BiasFeature]
		self.inputStates	=            [0] * (self.numInputs + 1)  # [ x_0, x_1, x_2, ..., x_{n-1},    1 ]
		self.weights		= np.asarray([0] * (self.numInputs + 1)) # [ w_0, w_1, w_2, ..., w_{n-1}, bias ]
		self.learningRate	= learningRate
		self.stateMin		= activationBounds[0]
		self.stateMax		= activationBounds[1]
		self.stateThreshold	= activationThreshold
		self.activation		= copy.copy(activation)
		self.dadz = copy.copy(dadz)
		self.dCda = copy.copy(dCda)
		self.stateReady = False
	def updateInputVector(self, features):
		if self.inputSources is None:
			# First layer
			self.inputStates = features + [1]
		else:
			# Get states from input neurons
			for i in range(self.numInputs):
				self.inputStates[i] = self.inputSources[i].classify(features)
	def calcPotential(self, features):
		"""
		TODO: documentation
		"""
		self.updateInputVector(features)
		# print("weights:  " + str(len(self.weights)))
		# print("features: " + str(len(self.inputStates)))
		self.potential = np.clip(np.dot(self.inputStates, self.weights), self.MIN_POTENTIAL, self.MAX_POTENTIAL)
	def checkThreshold(self):
		"""
		Outputs 1 if activation(potential) > (HIGH + LOW)/2
		"""
		if self.state > self.stateThreshold:
			return self.stateMax
		else:
			return self.stateMin
	def decision(self, features):
		"""
		Updates the state based on a feature vector.
		"""
		self.resetStateReady()
		self.calcPotential(features)
		self.state = self.activation(self.potential)
		self.setStateReady()
		return self.checkThreshold()
	def classify(self, features, continuous=False):
		"""
		Classifies a feature vector if given one; else calls getState() to return current state if available.
		features: features to classify.
		continuous: if True, return a value in the range [0, 1] instead of simply 0 or 1.
		"""
		if continuous:
			self.decision(features)
			return self.state
		else:
			return self.decision(features)
	def getState(self):
		"""
		Returns state if ready. If not, throws an error.
		"""
		if not self.stateReady:
			raise NotReadyException("State not ready.")
		return self.state
	def update(self):
		"""
		Updates weights and calls backprop() method of each input.
		"""
		self.resetStateReady()
		if self.inputSources is not None:
			for i in self.inputStates.keys():
				self.inputSources[i].backprop(self.weights[i] * self.error)
		self.weights = np.subtract(self.weights, np.multiply(np.asarray(self.inputStates), self.error * self.learningRate))
		# print(np.multiply(np.asarray(self.inputStates), self.error * self.learningRate))
		# print(self.weights)
		# print(self.inputStates)
		# print(str(self.error) + " " + str(self.learningRate))
	def learn(self, correctLabel):
		"""
		Update top layer.
		"""
		# print(self.potential)
		# print(str(self.dCda(self.state, correctLabel)) + " " + str(self.dadz(self.potential, self.state)))
		self.error = self.dCda(self.state, correctLabel) * self.dadz(self.potential, self.state)
		# if True: # abs(self.state - correctLabel) > 0.5:
		# 	print("    " + str(self.dCda(self.state, correctLabel)))
		# 	print("    " + str(self.dadz(self.potential, self.state)))
		self.update()
	def backprop(self, weightedError):
		"""
		Update layers other than the top layer via backpropagation.
		"""
		self.outputWeightedErrors += [weightedError]
		if len(self.outputWeightedErrors) == self.numOutputs:
			self.error = sum(self.outputWeightedErrors) * self.dadz(self.potential, self.state)
			self.outputWeightedErrors = []
			self.update()
	def setStateReady(self):
		"""
		Sets stateReady to True for this neuron (but not inputs).
		"""
		self.stateReady = True
	def resetStateReady(self):
		"""
		Sets stateReady to False for this neuron and all direct or indirect inputs.
		"""
		if self.stateReady:
			self.stateReady = False
			if self.inputSources is not None:
				for inputSource in inputSources:
					inputSource.resetStateReady()
	def getModelDict(self):
		"""
		Creates a dict sufficient for recreating this classifier.
		"""
		model = {
			'learningRate':	self.learningRate,
			'numInputs':	self.numInputs,
			'weights':		self.weights.tolist()
		}
		return model
	def saveToFile(self, fileName):
		"""
		Saves a representation sufficient for reconstruction to the indicated file.
		"""
		with open(fileName, "w") as fp:
			fp.write(str(self.getModelDict()))
	def fromModelDict(model):
		"""
		Reconstructs a classifier from the provided dict.
		"""
		binlr = BinaryLR(
			model['numInputs'],
			model['learningRate']
		)
		binlr.weights = np.asarray(model['weights'])
		return binlr
	def loadFromFile(fileName):
		"""
		Reconstructs a classifier from the indicated file.
		"""
		binlr = None
		with open(fileName, "r") as fp:
			binlr = BinaryLR.fromModelDict(literal_eval(fp.read()))
		return binlr