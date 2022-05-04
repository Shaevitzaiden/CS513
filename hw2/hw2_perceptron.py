#!/usr/bin/env python3

from tkinter import Y
from xml.sax.handler import feature_external_ges
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


class Perceptron_Basic: 
    def __init__(self, dev_set, dev_y, bin_map) -> None:
        self.weights = None
        self.dev_set = dev_set
        self.dev_y = dev_y
           
    def train(self, training_set, training_gt, epochs=5, lr=1):
        """ Train perceptron """
        bias_dimension = np.ones((training_set.shape[0],1))
        training_features = np.concatenate((training_set, bias_dimension), axis=1)

        self.weights = np.zeros((training_features.shape[1],))
        # self.weights = np.random.uniform(low=-1, high=1, size=(training_features.shape[1],))

        # Training loop
        for e in range(epochs):
            num_updates = 0
            for j in range(training_features.shape[0]):
                if training_gt[j]*np.dot(self.weights, training_features[j,:]) <= 0: # Take advange of mismatched signs meaning negative number will mean mismatches solutions
                    self.weights += lr*training_gt[j]*training_features[j,:]
                    num_updates += 1
            p_updates = num_updates / training_features.shape[0]
            p_correct, p_positive = self.eval(self.dev_set, self.dev_y)
            print("epoch {0} updates {1} ({2}% ) dev_error {3}% (+:{4}% )".format(e+1, num_updates, 
                round(p_updates*100,3), round((1-p_correct)*100,3), round(p_positive*100, 3)))
    
    def classify(self, ds):
        """ Classify a full set taking advantage of matrix multiplication"""
        bias_dimension = np.ones((ds.shape[0],1))
        ds = np.hstack([ds, bias_dimension])
        
        predictions = np.matmul(ds, self.weights).flatten()
        predictions = np.array([1 if (predictions[i]>=0) else -1 for i in range(predictions.size)])
        
        return predictions

    def eval(self, test_set, test_gt):
        """ Use numpy broadcasting to quickly determine percentage correct and percentage positve"""
        predictions = self.classify(test_set)
        correct = np.mean(test_gt == predictions)
        positive = np.mean(predictions > 0)
        return correct, positive

class Perceptron_Averaged: 
    def __init__(self, dev_set, dev_y, mapping) -> None:
        self.weights_s = None
        self.dev_set = dev_set
        self.dev_y = dev_y
        self.mapping = mapping
           
    def train(self, training_set, training_gt, epochs=5, lr=1):
        """ Train perceptron """
        bias_dimension = np.ones((training_set.shape[0],1))
        training_features = np.concatenate((training_set, bias_dimension), axis=1)

        self.weights_s = np.zeros((training_features.shape[1],))
        weights = np.zeros((training_features.shape[1],))

        # Training loop
        for e in range(epochs):
            num_updates = 0
            for j in range(training_features.shape[0]):
                if training_gt[j]*np.dot(weights, training_features[j,:]) <= 0: # Take advange of mismatched signs meaning negative number will mean mismatches solutions
                    weights += lr*training_gt[j]*training_features[j,:]
                    num_updates += 1
                self.weights_s += weights
            p_updates = num_updates / training_features.shape[0]
            p_correct, p_positive, _ = self.eval(self.dev_set, self.dev_y)
            print("epoch {0} updates {1} ({2}% ) dev_error {3}% (+:{4}% )".format(e+1, num_updates, 
                round(p_updates*100,3), round((1-p_correct)*100,3), round(p_positive*100, 3)))
        self.weights_s = self.weights_s / (e*training_features.shape[0])

    def weight_sorting(self):
        highest = np.flip(np.argsort(self.weights_s))[:5]
        for idx in highest:
            for key, value in self.mapping.items():
                if idx == value:
                    feature = key
            print("{0}  {1}".format(feature, self.weights_s[idx]))


    def classify(self, ds):
        """ Classify a full set taking advantage of matrix multiplication"""
        bias_dimension = np.ones((ds.shape[0],1))
        ds = np.hstack([ds, bias_dimension])
        
        predictions = np.matmul(ds, self.weights_s).flatten()
        predictions = np.array([1 if (predictions[i]>=0) else -1 for i in range(predictions.size)])
        
        return predictions

    def eval(self, test_set, test_gt, gt=True):
        """ Use numpy broadcasting to quickly determine percentage correct and percentage positve"""
        predictions = self.classify(test_set)
        correct = None
        if gt:
            correct = np.mean(test_gt == predictions)
        positive = np.mean(predictions > 0)
        return correct, positive, predictions


def make_binary_mapping(data):
    # create binarization mapping where "new_data" is lists of integers corresponding to hot indices in the binarized matrix
    mapping = {}
    new_data = []
    for row in data:
        new_row = []
        for j, x in enumerate(row[:-1]):
            feature = (j, x) # j is the column index and x is the value
            if feature not in mapping: # new feature
                mapping[feature] = len(mapping) # insert a new feature into the index
            new_row.append(mapping[feature])

    return mapping

def binarize(data, mapping, numerical=False):
    num_features = len(mapping)
    binarized_features = np.zeros([len(data), num_features])
    outputs = np.zeros([len(data)],dtype=np.int64)
    age_list = []
    hours_list = []
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            if j == (len(row)-1):
                outputs[i] = (1 if x==">50K" else -1)
            else:
                try:
                    binarized_features[i][mapping[j, x]] = 1
                except KeyError:
                    pass
        if numerical:
            age_list.append(scale_age(int(row[0])))
            hours_list.append(scale_hours(int(row[7])))

    if numerical:
        numerical_fields = np.vstack((np.array([age_list]),np.array([hours_list]))).T
        binarized_features = np.hstack((binarized_features, numerical_fields))
    return binarized_features, outputs

def feature_engineering(data_set, feature_set, zero_mean=False, unit_variance=False):
    for feature in feature_set:

        mean = np.mean(data_set[:,feature],0) 
        std  = np.std(data_set[:,feature],0)
    
        if zero_mean:
            data_set[:,feature]= data_set[:,feature] - mean
        if unit_variance:
            data_set = data_set[:,feature] / std
    return data_set


def sort_features(feature_set, output_set, first='neg'):
    full_set = np.hstack((feature_set, output_set.reshape((5000,1))))
    full_set = full_set[full_set[:, -1].argsort()]
    if first == "neg":
        return full_set[:,:-1], full_set[:,-1]
    elif first == "pos":
        return full_set[:,:-1], np.flip(full_set[:,-1])   

def scale_age(age):
    a = 2 / (73)
    b = -a * 17
    val = a*age + b
    return val

def scale_hours(hours):
    a = 2 / (98)
    b = -a 
    val = a*hours + b
    return val


if __name__ == "__main__":
    # Open text file    
    lines_train = open('income.train.txt.5k').readlines()
    data_train = [line.strip().split(", ") for line in lines_train]

    lines_dev = open('income.dev.txt').readlines()
    data_dev = [line.strip().split(", ") for line in lines_dev]

    lines_test = open('income.test.blind').readlines()
    data_test = [line.strip().split(", ") for line in lines_test]
    
    # Create binary mapping from training data
    bin_map = make_binary_mapping(data_train)
    
    # Gets binary features from datasets
    t1 = time.time()
    bindata_dev, outputs_dev = binarize(data_dev, bin_map,numerical=False)
    bindata_train, outputs_train = binarize(data_train, bin_map,numerical=False)

    
    
    p = Perceptron_Averaged(bindata_dev, outputs_dev, bin_map)
    p.train(bindata_train, outputs_train, epochs=3, lr=1)
    
    correct, positive, predictions = p.eval(bindata_dev, outputs_dev, gt=False)
    for i, p in enumerate(predictions):
        label = ">50K" if p==1 else "<=50K" #
        print(", ".join(data_test[i] + [label])) # output 10 fields, separated by ", "