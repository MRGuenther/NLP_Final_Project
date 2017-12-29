The program requires the numpy package. It can be installed with the following command:
	pip install numpy

Usage:
	python s_split.py train trainingFileName numEpochs modelOutputFileName [learningRate [modelInputFileName]]
	python s_split.py test testingFileName modelInputFileName
	python s_split.py vote testingFileName votingThreshold modelInputFileName_0 [...]
	python s_split.py bootstrap testingFileName numSamples baseline votingThreshold modelInputFileName_0 [...]

The train command takes a training set's file path, the desired number of training epochs, and the desired output file path. Optionally, it can take a learning rate (which defaults to 0.1 for new models) and the path of a model to load for further training.

The test command takes the paths to the testing data set and the model to be tested.

The vote command takes the path to the testing data set, the number of votes required for a positive (split) classification, and then an arbitrary number of model file paths to load into the voting ensemble.

The bootstrap command takes the testing file path, the number of samples to take, and then the voting threshold and input models as per the vote command.



Note:
The file data/splitDataIntoSubsets.py is simply a utility I wrote to split the single data set I had been given into three sets (for training, tuning, and testing, respectively). All the program does is split a .csv file into an arbitrary number of equal (or as close as possible to equal) partitions. I never used data/OnlyOneSimplifiedVersion.csv except to generate those partitions of it; it is included here only for completeness.