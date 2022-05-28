#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_log_error
import sklearn as sk
import sys
import time


# def make_binary_mapping(data):
#     # create binarization mapping where "new_data" is lists of integers corresponding to hot indices in the binarized matrix
#     mapping = {}
#     new_data = []
#     for row in data:
#         new_row = []
#         for j, x in enumerate(row[:-1]):
#             feature = (j, x) # j is the column index and x is the value
#             if feature not in mapping: # new feature
#                 mapping[feature] = len(mapping) # insert a new feature into the index
#             new_row.append(mapping[feature])
#     return mapping


def make_binary_mapping(data, numeric=[]):
    # create binarization mapping where "new_data" is lists of integers corresponding to hot indices in the binarized matrix
    mapping = {}
    new_data = []
    for row in data:
        new_row = []
        for j, x in enumerate(row[:-1]):
            if j in numeric:
                feature = (j, "numeric") # Create placeholder for numerical  values in mapping
            else:
                feature = (j, x) # j is the column index and x is the value
            if feature not in mapping: # new feature
                mapping[feature] = len(mapping) # insert a new feature into the index
            new_row.append(mapping[feature])
    return mapping


def binarize(data, mapping, numeric=[], nonnlinear=[], combos=[], test=False):
    num_features = len(mapping) + len(combos)
    num_samples = len(data)
    feature_map = np.zeros([num_samples, num_features],dtype=np.int64)
    outputs = np.zeros([len(data)],dtype=np.int64)
    nonlin_feats = [x[0] for x in nonnlinear]
    nonlin_funcs = [x[1] for x in nonnlinear]

    for i, row in enumerate(data):
        for j, x in enumerate(row):
            if j == (len(row)-1) and not test:
                outputs[i] = int(row[-1])
            elif j in numeric:
                try:
                    if j in nonlin_feats:
                        f = nonlin_funcs[0]
                        feature_map[i][mapping[j,'numeric']] = f(int(x))
                        nonlin_funcs.append(nonlin_funcs.pop(0))
                    else:
                        feature_map[i][mapping[j,'numeric']] = int(x)
                except ValueError:
                    feature_map[i][mapping[j,'numeric']] = 0
            else:
                try:
                    feature_map[i][mapping[j, x]] = 1
                except KeyError:
                    pass
    j = len(mapping)
    # Creates feature combinations   
    for k, (f1, f2, type) in enumerate(combos):
        for z, row in enumerate(data):
            if type == "s":
                feature_map[z][k+j] = f1 - f2
            elif type == "m":
                feature_map[z][k+j] = f1 * f2

    return feature_map, outputs

def sort_features(weights, binary_mapping, field_names, num_features=10, first='pos'):
    weights_idx = np.argsort(weights)
    # weights_idx = np.flip(weights_idx)
    
    key_features = []
    for i in range(num_features):
        for key in binary_mapping.keys():
            if binary_mapping[key] == weights_idx[i]:
                x = (key[0], field_names[key[0]], weights[weights_idx[i]])
                key_features.append(x)
    return key_features



if __name__ == "__main__":
    # Load in training data -------------------------------
    lines_train = open('hw3-data/my_train.csv').readlines()
    data_train = [line.strip().split(",") for line in lines_train]
    
    # Remove first row containing field names and first column containing id
    field_names = data_train.pop(0)
    for row in data_train:
        _ = row.pop(0)

    # Load in dev data ---------------------------------------
    lines_dev = open('hw3-data/my_dev.csv').readlines()
    data_dev = [line.strip().split(",") for line in lines_dev]

    # Remove first row containing field names and first column containing id
    field_names = data_dev.pop(0)
    for row in data_dev:
        _ = row.pop(0)
    
    # Load in test data ---------------------------------------
    lines_test = open('hw3-data/test.csv').readlines()
    data_test = [line.strip().split(",") for line in lines_test]
    field_names = data_test.pop(0)
    field_names.pop(0)
    submission_ids = []
    for row in data_test:
        submission_ids.append(row.pop(0))    

    # ------------------ Field additions -------------------------
    numeric_fields = [2, 3, 16, 17, 18, 19, 25, 33, 35, 36, 37, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
                    51, 53, 55, 58, 60, 61, 65, 66, 67, 68, 69, 70, 74, 75, 76]
    mixed_fields =[2, 58, 33, 35, 36, 37, 61, 25, 46, 47, 60]
    numeric_fields = []
    mixed_fields = []


    quad = lambda x: x**2
    nlog = lambda x: np.log(x)
    nonlin = [] #[(3, nlog)]
    # yearbuilt-yearsold,  overalqual*overallcond
    combos = [] #[(18, 76, "s"), (16, 17, "m")] #, (12, 13, 'm')]

    # for i in range(len(field_names)):
    #     print("{0}, {1}".format(i, field_names[i]))


    # Create binary mappings from  data
    bin_map = make_binary_mapping(data_train,numeric=numeric_fields)

    # Create feature maps
    bindata_train, outputs_train = binarize(data_train, bin_map, numeric=numeric_fields,nonnlinear=nonlin,combos=combos)
    bindata_dev, outputs_dev = binarize(data_dev, bin_map, numeric=numeric_fields,nonnlinear=nonlin,combos=combos)
    bindata_test, _ = binarize(data_test, bin_map, numeric=numeric_fields,nonnlinear=nonlin, test=True,combos=combos)
    reshaped_test_output = outputs_train.reshape((bindata_train.shape[0],1))
    
    # Create and fit model
    model = linear_model.LinearRegression()
    # model = linear_model.Ridge(alpha=20)

    model.fit(bindata_train, np.log(reshaped_test_output))
    
    # -------------- Dev set --------------
    price_predict = model.predict(bindata_dev)
    print(np.sqrt(mean_squared_log_error(outputs_dev, np.exp(price_predict))))
    bias = model.intercept_
    coeffs = model.coef_
    coeffs_idx = np.argsort(coeffs)
    
    # ------------- Find top and bottom 10 ----------
    sorted_features = sort_features(coeffs[0,:], bin_map, field_names)
    print("Col, Field,  Weight")
    for feature in sorted_features:
        print(feature)
    

    # -------------- Test set -------------
    

    price_predict = model.predict(bindata_test)
    
    # print("Id,SalePrice")
    # for i in range(price_predict.shape[0]):
    #     print(",".join([submission_ids[i]] + [str(np.exp(price_predict[i,0]))])) 