#!/usr/bin/env python3

from __future__ import division

from svector import svector
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB

import sys
import time
import numpy as np


def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_cache(textfile, vectorize=True):  
    label_words = []
    for line in open(textfile):
        label, words = line.strip().split("\t")
        label_words.append((1 if label=="+" else -1, make_vector(words.split()) if vectorize else words.split()))
    return label_words       

def make_pruned_file(textfile,singles=True,doubles=False,others=[]):
    all_words = []
    label_words = []
    singles_list = []
    doubles_list = []

    # get all words into list
    for line in open(textfile):
        label, words = line.strip().split("\t")
        words = words.split()
        label_words.append((label,words))
        all_words+=words
    

    # Find unique words and their counts, fill list with words with only one occurrence
    uniq_words = set(all_words)
    for word in uniq_words:
        if singles and all_words.count(word) == 1 and singles:
            singles_list.append(word)
        if all_words.count(word) == 2 and doubles:
            doubles_list.append(word)
    
    title = ['pruned']
    if singles: title += ['_singles']
    if doubles: title += ['_doubles']
    if others != []: title += ['_others']
    title+=['.txt']
    title = "".join(title)

    with open (title, 'w') as file:
        for i, (label,words) in enumerate(label_words):
            for word in words:
                if word in singles_list and singles:
                    words.remove(word)
                if word in doubles_list and singles:
                    words.remove(word)
                if word in others:
                    words.remove(word)
            if len(words) != 0:
                file.write("{0}\t{1}\n".format(label, " ".join(words)))

def write_predictions_to_file(labels, label_words):
    with open ("hw4-data\\test.txt.predicted", 'w') as file:
        for i, (label,words) in enumerate(label_words):
            file.write("{0}\t{1}\n".format('+' if labels[i]==1 else "-", " ".join(words)))

def make_vector(words):
    v = svector()
    v['<bias>'] = 1
    for word in words:
        v[word] += 1
    return v
    
def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def test2(devfile, model):
    tot, err = 0, 0
    x = []
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        prediction = (model.dot(make_vector(words))) 
        if label * prediction <= 0: 
            err += 1
            x.append((prediction, words))
    x_sorted = sorted(x)
    
    print("Should have been labeled positive")
    for i, (key, val) in enumerate(x_sorted[0:5], start=1):
        print("{0}: {1}, {2}".format(i, key, val))

    print("Should have been labeled negative")
    for i, (key, val) in enumerate(x_sorted[-5:], start=1):
        print("{0}: {1}, {2}".format(i, key, val))
            
def test3(dev_cache, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(dev_cache, 1): # note 1...|D|
        err += label * (model.dot(words)) <= 0
    return err/i  # i is |D| now

def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
        dev_err = test(devfile, model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

def train_averaged(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    c = 0
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_a += c*label*sent
            c += 1
        test_model = c*model - model_a
        dev_err = test(devfile, test_model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return test_model

def train_averaged_w_cache(trainfile, devfile, epochs=5):
    cached_train = make_cache(trainfile)
    cached_dev = make_cache(devfile)
    t = time.time()
    best_err = 1.
    model = svector()
    model_a = svector()
    c = 0
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, sent) in enumerate(cached_train, 1): # label is +1 or -1
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                model_a += c*label*sent
            c += 1
        test_model = c*model - model_a
        dev_err = test3(cached_dev, test_model)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))
    return test_model

def find_features(model):
    model_vals = [(val,key) for key,val in model.items()]
    sorted_model_vals = sorted(model_vals)
    print("Most Negative")
    for i, (key, val) in enumerate(sorted_model_vals[0:20], start=1):
        print("{0}: {1}, {2}".format(i, key, val))

    print("Most Positive")
    for i, (key, val) in enumerate(sorted_model_vals[-20:], start=1):
        print("{0}: {1}, {2}".format(i, key, val))

def build_mapping(labeled_words):
    # mapping = {'<bias>': 1}
    mapping = {}
    idx = 0
    for (label, words) in labeled_words:
        for word in words:
            try:
                _ = mapping[word]
            except KeyError:
                mapping[word] = idx
                idx += 1
    return mapping

def build_array_from_mapping(mapping, labeled_words, test=False):
    X = np.zeros((len(labeled_words), len(mapping)),dtype=np.int32)
    Y = np.zeros((len(labeled_words),),dtype=np.int32)
    for row, lw in enumerate(labeled_words):
        for word in lw[1]:
            try:
                X[row,mapping[word]] += 1
            except KeyError:
                pass
        if not test:
            Y[row] = int(lw[0])
    return X, Y
    
def trainMLP(mlpModel, x, y, xd, yd, epochs, tol=0):
    for i in range(epochs):
        mlpModel.fit(x, y)
        p = mlpModel.predict(xd)
        correct = p == yd
        e = (1-np.sum(correct)/yd.shape[0])*100
        print("epoch: {0},  error: {1}".format(i+1, e))
    return mlpModel



if __name__ == "__main__":
    # train(sys.argv[1], sys.argv[2], 10)
    train_file = "hw4-data\\pruned_singles_others.txt"
    dev_file = "hw4-data\\dev.txt"
    test_file = "hw4-data\\test.txt"
    # model = train_averaged(train_file, dev_file, 10)
    
    stop_words = ["the","and","an","in","a","because","or","it","my","I","your",
                "mine","at","am","as","can","by","do","did","for"]
    
    # Function to make new text files with various pruning
    # make_pruned_file("hw4-data\\train.txt",singles=True, doubles=False, others=stop_words)
    
    # model = train_averaged_w_cache("pruned_singles.txt", dev_file, 10)
    
    lw_train = make_cache(train_file, vectorize=False)
    lw_dev = make_cache(dev_file, vectorize=False)
    lw_test = make_cache(test_file, vectorize=False)

    # Build mapping and numpy arrays for training sklearn models
    mapping = build_mapping(lw_train)
    x_train, y_train = build_array_from_mapping(mapping, lw_train)
    x_dev, y_dev = build_array_from_mapping(mapping, lw_dev)
    x_test, y_test = build_array_from_mapping(mapping, lw_test)

    # model = SGDClassifier()
    # model = MLPClassifier(max_iter=1, warm_start=True, activation='tanh', learning_rate_init=0.0005)
    # model = trainMLP(model, x_train, y_train, x_dev, y_dev, epochs=20)
    # model = BernoulliNB()
    # model = MultinomialNB()
    model = ComplementNB(alpha=1.08)
    model.fit(x_train, y_train)   

    predictions = model.predict(x_test)
    write_predictions_to_file(predictions, lw_test)

    # correct = predictions == y_dev
    # e = (1-np.sum(correct)/y_dev.shape[0])*100
    # print("Dev-Set error: {0}".format(e))