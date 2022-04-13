#!/usr/bin/env python3

import numpy as np
import sys


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

def make_binary_mapping(data, numerical_features=[0,7]):
    # create binarization mapping where "new_data" is lists of integers corresponding to hot indices in the binarized matrix
    mapping = {}
    new_data = []
    for row in data[:-1]:
        new_row = []
        for j, x in enumerate(row):
            if j in numerical_features:
                feature = (j, "numerical") # Create placeholder for numerical  values in mapping
            else:
                feature = (j, x) # j is the column index and x is the value
            if feature not in mapping: # new feature
                mapping[feature] = len(mapping) # insert a new feature into the index
            new_row.append(mapping[feature])
    return mapping

def binarize(data, mapping, numerical_features=[0,7]):
    num_features = len(mapping) + len(numerical_features)
    binarized_features = np.zeros([len(data), num_features])
    outputs = []
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            try:
                if j == 0: # Skip the age, hours worked, and maybe the pos/neg outcome
                    binarized_features[i, j] = scale_age(float(x))
                elif j == 7:
                    binarized_features[i, j] = scale_hours(float(x))
                elif j == 9:
                    outputs.append(1 if x==">50K" else -1)
                else:
                    try:
                        binarized_features[i][mapping[j, x]] = 1
                    except KeyError:
                        pass
            except IndexError: # catches case where data doesn't have the 10th(9th) element, namely the training data
                print(j)
                pass
    return binarized_features, outputs

def knn(example_feature, train_features, train_output, k=3, o=2):
    """ k-nearest-neighbor classifier
    :params ndarray example_features: binarized array of features to be classified
    :params ndarray train_features: binarized array of features to be compared against
    :params list train_output: binarized array to find classification of example
    :params int k: number of nearest neighbors to evaluate
    :params int o: 0 (euclidean distance), 1 (manhattan distance)
    """
    try:
        neighbors = np.argpartition(np.linalg.norm(example_feature - train_features, axis=1, ord=o), k)[:k]
        # neighbors = np.argsort(np.linalg.norm(example_features - train_features, axis=1, ord=o))[:k]
        # for i in range(k):
        #     print(data[neighbors[i]])
        votes = sum([train_output[neighbors[i]] for i in range(k)])
    except ValueError:
        votes = sum(train_output) # If the number of nearest neighbors exceeds the length of the training data, sum the training data votes
    prediction = 1 if votes > 0 else -1
    return prediction

def knn_eval(example_features, example_output, train_features, train_output, k=3, o=2):
    error = 0
    positive = 0
    predictions = []
    for i, feature in enumerate(example_features):
        prediction = knn(feature, train_features, train_output, k, o)
        if prediction != example_output[i]:
            error += 1
        if prediction == 1:
            positive += 1
        predictions.append(prediction)
    percent_error = error / len(example_output) * 100
    percent_positive = positive / len(example_output) * 100
    return percent_error, percent_positive, predictions
    

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
    bindata_dev, outputs_dev = binarize(data_dev, bin_map)
    bindata_train, outputs_train = binarize(data_train, bin_map)
    bindata_test, outputs_test = binarize(data_test, bin_map) # outputs empty list for test set
    outputs_test = [i for i in range(bindata_test.shape[0])] # Just to get code to run
    
    error, pos, predictions = knn_eval(bindata_test, outputs_test, bindata_train, outputs_train, k=9, o=2)

    for i, p in enumerate(predictions):
        label = ">50K" if p==1 else "<=50K" #
        print(", ".join(data_test[i] + [label])) # output 10 fields, separated by ", "
    
    # ks = [1,3,5,7,9,99,999,9999]
    # for k in ks:
    #     error, pos, predictions = knn_eval(bindata_dev, outputs_dev, bindata_train, outputs_train, k=k, o=2)
    #     print("k={0}    dev_error {1}% (+:{2}% )".format(k, round(error,1),round(pos,1)))
    


    # # Get closed people by euclidean and manhattan distance
    # last_person_dev = bindata_dev[-1,:]
    # print(data_dev[-1])

    # euclid_distances = np.linalg.norm(bindata_train-last_person_dev, axis=1)
    # manhattan_distances = np.linalg.norm(bindata_train-last_person_dev, axis=1, ord=1)

    # idx = np.argsort(euclid_distances)
    # print(idx[0:5])
    # for i in idx[0:5]:
    #     print(euclid_distances[i])
    #     print(data_train[i])