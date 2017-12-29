import sys
import re

tagPtn = re.compile("\((\S+)")

def main(fileName):
	tagSet = set()
	with open(fileName, "r", encoding="utf8") as fp:
		content = fp.read()
	for match in tagPtn.finditer(content):
		tag = match.group(1)
		if tag not in tagSet:
			tagSet.add(tag)
	print(tagSet)
	return tagSet

if __name__ == "__main__":
	main(sys.argv[1])