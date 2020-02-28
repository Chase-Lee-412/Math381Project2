import os
import nltk.data

f = open("data/Trump.txt", "r")
trainFile = f.readlines()
os.remove("data/output.txt")
f = open("data/output.txt", "w", newline='\n')



tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

fp = open("data/Trump.txt")

data = fp.read()

print('\n-----\n'.join(tokenizer.tokenize(data)))

# for lines in trainFile:
#     lines = nltk.tokenize.sent_tokenize(lines)
#     for line in lines:
#         f.write(line)