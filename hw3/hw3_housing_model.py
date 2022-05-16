#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_log_error
import sklearn as sk
import sys
import time


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

def binarize(data, mapping, numerical=False, test=False):
    num_features = len(mapping)
    binarized_features = np.zeros([len(data), num_features],dtype=np.int64)
    outputs = np.zeros([len(data)],dtype=np.int64)
    age_list = []
    hours_list = []
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            if j == (len(row)-1) and not test:
                outputs[i] = int(row[-1])
            else:
                try:
                    binarized_features[i][mapping[j, x]] = 1
                except KeyError:
                    pass
        if numerical:
            # age_list.append(scale_age(int(row[0])))
            # hours_list.append(scale_hours(int(row[7])))
            pass

    if numerical:
        numerical_fields = np.vstack((np.array([age_list]),np.array([hours_list]))).T
        binarized_features = np.hstack((binarized_features, numerical_fields))
    return binarized_features, outputs


def sort_features(feature_set, output_set, first='neg'):
    full_set = np.hstack((feature_set, output_set.reshape((5000,1))))
    full_set = full_set[full_set[:, -1].argsort()]
    if first == "neg":
        return full_set[:,:-1], full_set[:,-1]
    elif first == "pos":
        return full_set[:,:-1], np.flip(full_set[:,-1])   


if __name__ == "__main__":
    # Load in training data
    lines_train = open('hw3-data/my_train.csv').readlines()
    data_train = [line.strip().split(",") for line in lines_train]
    
    # Remove first row containing field names and first column containing id
    field_names = data_train.pop(0)
    for row in data_train:
        row.pop(0)
    
    lines_dev = open('hw3-data/my_dev.csv').readlines()
    data_dev = [line.strip().split(",") for line in lines_dev]

    # Remove first row containing field names and first column containing id
    field_names = data_dev.pop(0)
    for row in data_dev:
        row.pop(0)

    lines_test = open('hw3-data/test.csv').readlines()
    data_test = [line.strip().split(",") for line in lines_test]
    field_names = data_test.pop(0)
    submission_ids = []
    for row in data_test:
        submission_ids.append(row.pop(0))    


    # Create binary mapping from training data
    bin_map = make_binary_mapping(data_train)
    bindata_train, outputs_train = binarize(data_train, bin_map, numerical=False)
    bindata_dev, outputs_dev = binarize(data_dev, bin_map, numerical=False)
    bindata_test, _ = binarize(data_test, bin_map, numerical=False, test=True)

    bindata_train = np.array(bindata_train, dtype=np.int64)
    reshaped_output = outputs_train.reshape((bindata_train.shape[0],1))
    regr = linear_model.LinearRegression()
    
    regr.fit(bindata_train, np.log(reshaped_output))

    price_predict = regr.predict(bindata_test)
    
    print("Id,SalePrice")
    for i in range(price_predict.shape[0]):
        print(",".join([submission_ids[i]] + [str(np.exp(price_predict[i,0]))])) 