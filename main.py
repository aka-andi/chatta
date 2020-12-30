import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# stemming with tokenization
for intent in data["intents"]:
    for pattern in intent[""]:  
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w.isalpha()]
words = sorted(list(set(words)))

labels = sorted(labels)

# bag of words one hot encoding for training model
training = []
out = []

out_empty = [0 for _ in range(len(labels))] # _ throwaway variable

for x, doc in enumerate(docs_x):
    bag = []

    tokens = [stemmer.stem(w) for w in doc]

    for w in words:
        status = 1 if w in tokens else 0
        bag.append(status)
    
    out_row = out_empty[:]  # initialize copy
    out_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    out.append(out_row)

training = numpy.array(training)
out = np.array(out)