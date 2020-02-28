import math
from collections import Counter
from typing import List
import os


f = open("data/Trump.txt", "r")
trainFile = f.readlines()
f = open("outputBigram.txt", "w", newline='\n')

def setUNK(count: Counter, unkcutoff: int) -> None:
    count["::UNK::"] = 0
    for x in count.copy():
        if count[x] <= unkcutoff:
            count["::UNK::"] += count[x]
            del count[x]

def UnigramCounterFunction(file: List[str], unkcutoff: int) -> Counter:
    UnigramCount = Counter()
    for x in file:
        x = x.split()
        # Inserting ::STOP:: for end of sentence
        x.insert(len(x),"::STOP::")
        for word in x:
            UnigramCount[word] += 1
    #Dealing with low frequency. Adding UNK
    setUNK(UnigramCount, unkcutoff)
    return UnigramCount

def BigramCounterFunction(file: List[str], Unigram: Counter, unkcutoff: int) -> Counter:
    BigramCount = Counter()
    numWords = 0
    for x in file:
        x = x.split()
        # Inserting ::START:: for convenience
        x.insert(0,"::START::")
        # Inserting ::STOP:: for end of sentence
        x.insert(len(x),"::STOP::")
        # Changing words I haven't seen to UNK
        for i in range(1,len(x)):
            if x[i] not in Unigram:
                x[i] = "::UNK::"
        for i in range(1,len(x)):
            word = (x[i-1],x[i])
            BigramCount[word] += 1
    return BigramCount

def perplexity(Unigram: Counter, Bigram: Counter, file: List[str], k: float) -> float:
    score = 0
    M = 0
    sizeOfVocab = len(Unigram)
    for x in file:
        x = x.split()
        x.insert(0,"::START::")
        x.insert(len(x), "::STOP::")
        # Changing words I haven't seen to UNK
        for i in range(1,len(x)):
            if x[i] not in Unigram:
                x[i] = "::UNK::"
        for i in range(1,len(x)):
            twoWords = (x[i-1],x[i])
            M += 1
            if i == 1:
                score += math.log((Bigram[twoWords]+k)/(Unigram["::STOP::"]+sizeOfVocab*k),2)
            else:    
                score += math.log((Bigram[twoWords]+k)/(Unigram[x[i-1]]+sizeOfVocab*k),2)
    score = score/M
    return math.pow(2,-score)

def main():
    cutoff = 1
    kSet = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    Unigram = UnigramCounterFunction(trainFile, cutoff)
    Bigram = BigramCounterFunction(trainFile, Unigram, cutoff)
    print("Trained the model with Small brown.train.txt.")
    for k in kSet:
        print("k value: ", k)
        print("perplexity is : ", perplexity(Unigram, Bigram, trainFile, k))

main()
