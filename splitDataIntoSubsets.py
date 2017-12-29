import sys

def main(inputFileName, outputFileNames):
	numOutputs = len(outputFileNames)
	with open(inputFileName, "r", encoding="utf8") as fin:
		lines = fin.read().split("\n")
	firstLine = lines[0]
	lines = lines[1:]
	numLines = len(lines)
	lastBreak = 0
	for i in range(numOutputs):
		nextBreak = lastBreak + int((numLines - lastBreak) / (numOutputs - i))
		with open(outputFileNames[i], "w", encoding="utf8") as fout:
			fout.write("\n".join([firstLine] + lines[lastBreak:nextBreak]))
		lastBreak = nextBreak

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2:])