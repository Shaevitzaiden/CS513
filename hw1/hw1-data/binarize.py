#!/usr/bin/env python3

import numpy as np


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
    outcomes = []
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            if j == 0: # Skip the age, hours worked, and maybe the pos/neg outcome
                binarized_features[i, j] = scale_age(float(x))
            elif j == 7:
                binarized_features[i, j] = scale_hours(float(x))
            elif j == 9:
                outcomes.append(1 if x==">50K" else 0)
            else:
                try:
                    binarized_features[i][mapping[j, x]] = 1
                except KeyError:
                    pass
    return binarized_features, outcomes

def knn(example_features, train_features, train_output, k=3, o=0):
    """ k-nearest-neighbor classifier
    :params ndarray example_features: binarized array of features to be classified
    :params ndarray train_features: binarized array of features to be compared against
    :params ndarray train_output: binarized array to find classification of example
    """
    
    neighbors = np.argpartition(np.linalg.norm(example_features - train_features, axis=1, ord=o), k)[:k]



if __name__ == "__main__":
        # Open text file    
    lines_train = open('income.train.txt.5k').readlines()
    data_train = [line.strip().split(", ") for line in lines_train]

    lines_dev = open('income.dev.txt').readlines()
    data_dev = [line.strip().split(", ") for line in lines_dev]
    
    
    bin_map = make_binary_mapping(data_train)
    bindata_dev, outcomes_dev = binarize(data_dev, bin_map)
    bindata_train, outcomes_train = binarize(data_train, bin_map)


    # Get closed people by euclidean and manhattan distance
    last_person_dev = bindata_dev[-1,:]
    print(data_dev[-1])

    euclid_distances = np.linalg.norm(bindata_train-last_person_dev, axis=1)
    manhattan_distances = np.linalg.norm(bindata_train-last_person_dev, axis=1, ord=1)

    idx = np.argsort(euclid_distances)
    print(idx[0:5])
    for i in idx[0:5]:
        print(euclid_distances[i])
        print(data_train[i])