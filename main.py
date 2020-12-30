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