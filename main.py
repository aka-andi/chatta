import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle # load lists into pickle file for subsequent training executions

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, out = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # stemming with tokenization
    for intent in data["intents"]:
        for pattern in intent["patterns"]:  
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

    training = np.array(training)
    out = np.array(out)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, out), f)

# training model using tflearn

nk = tflearn.input_data(shape=[None, len(training[0])])
nk = tflearn.fully_connected(nk, 8) # 8 neurons in first hidden layer
nk = tflearn.fully_connected(nk, 8) # 8 neurons in second hidden layer
nk = tflearn.fully_connected(nk, len(out[0]), activation="softmax") # output layer, neurons per label
nk = tflearn.regression(nk)

model = tflearn.DNN(nk)

try:
    model.load("model.tflearn")
except:
    model.fit(training, out, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("this is your bot speaking")
    while True:
        inpt = input("Me: ")
        if inpt.lower() == "quit":
            break
        
        out = model.predict([bag_of_words(inpt, words)])[0]
        tag = labels[np.argmax(out)]

        if out[np.argmax(out)] > 0.72:
            for intent in data["intents"]:
                if intent['tag'] == tag:
                    responses = intent['responses']
            print(random.choice(responses))
        else:
            print("I am not sure, can you be more specific please?")

chat()