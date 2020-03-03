import math
from collections import Counter
from typing import List, Tuple, Dict, Callable
import os
import numpy as np


f = open("data/Trump.txt", "r")
trainFile = f.readlines()

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

def generateTransitionMatrix(Bigram: Counter) -> Tuple[np.ndarray, List[str]]:
    words = set()
    for (w1, w2) in Bigram:
        words.add(w1)
        words.add(w2)
    words = list(words)
    words.sort()
    transition_matrix = np.zeros((len(words),len(words)))
    for ((w1, w2),val) in Bigram.items():
        transition_matrix[words.index(w1), words.index(w2)] = val
    row_sums = np.sum(transition_matrix, axis=1)
    rows_with_zero = np.nonzero(row_sums == 0)
    for [ind] in rows_with_zero:
        transition_matrix[ind, ind] = 1
        row_sums[ind] = 1
    transition_matrix / row_sums[:, None]
    return (transition_matrix, words)


def train(trainFile: List[str], cutoff: int, k) -> Callable[[List[str]], float]:
    Unigram = UnigramCounterFunction(trainFile, cutoff)
    Bigram = BigramCounterFunction(trainFile, Unigram, cutoff)
    return lambda test: perplexity(Unigram, Bigram, test, k)

def classifiy():
    train_set = range(1,46)
    test_set = range(46,51)
    fox_base_dir = "data/FoxNews/foxArticle"
    cnn_base_dir = "data/CnnNews/CNN-"
    fox_train = []
    cnn_train = []
    for i in train_set:
        with open(fox_base_dir + f"{i}.txt") as text_file:
            fox_train.extend(text_file.readlines())
        with open(cnn_base_dir + f"{i}.txt") as text_file:
            cnn_train.extend(text_file.readlines())
    fox_test = []
    cnn_test = []
    for i in test_set:
        with open(fox_base_dir + f"{i}.txt") as text_file:
            fox_test.append(text_file.readlines())
        with open(cnn_base_dir + f"{i}.txt") as text_file:
            cnn_test.append(text_file.readlines())
    cutoff = 2
    k = 0.1
    fox_perplexity = train(fox_train, cutoff, k)
    cnn_perplexity = train(cnn_train, cutoff, k)
    for i,text in enumerate(fox_test):
        print(f"Fox article {i+1} classified as {'fox' if fox_perplexity(text) < cnn_perplexity(text) else 'cnn'}")
    for i,text in enumerate(cnn_test):
        print(f"CNN article {i+1} classified as {'fox' if fox_perplexity(text) < cnn_perplexity(text) else 'cnn'}")


def main():
    cutoff = 1
    kSet = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    Unigram = UnigramCounterFunction(trainFile, cutoff)
    Bigram = BigramCounterFunction(trainFile, Unigram, cutoff)
    print("Trained the model with Small brown.train.txt.")
    for k in kSet:
        print("k value: ", k)
        print("perplexity is : ", perplexity(Unigram, Bigram, trainFile, k))



def exportTransitionMatrix():
    cutoff = 1
    kSet = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    Unigram = UnigramCounterFunction(trainFile, cutoff)
    Bigram = BigramCounterFunction(trainFile, Unigram, cutoff)
    return generateTransitionMatrix(Bigram)

classifiy()
exportTransitionMatrix()
