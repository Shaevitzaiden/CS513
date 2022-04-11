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

def make_binary_mapping(data):
    # create binarization mapping where "new_data" is lists of integers corresponding to hot indices in the binarized matrix
    mapping = {}
    new_data = []
    for row in data:
        new_row = []
        new_age_hour_row = []
        for j, x in enumerate(row):
            if j == 0: # Skip the age, hours worked, and maybe the pos/neg outcome
                continue
            elif j == 7:
                continue
            elif j == 9:
                continue
            else:
                feature = (j, x) # j is the column index and x is the value
                if feature not in mapping: # new feature
                    mapping[feature] = len(mapping) # insert a new feature into the index
                new_row.append(mapping[feature])
    return mapping

def binarize(data, mapping):
    new_data = []
    new_age_hours = []
    for row in data:
        new_row = []
        new_age_hour_row = []
        for j, x in enumerate(row):
            if j == 0: # Skip the age, hours worked, and maybe the pos/neg outcome
                new_age_hour_row.append(scale_age(int(x)))
                continue
            elif j == 7:
                new_age_hour_row.append(scale_hours(int(x)))
                continue
            elif j == 9:
                continue
            else:
                feature = (j, x) # j is the column index and x is the value
                try:
                    new_row.append(mapping[feature])
                except KeyError:
                    pass
        new_data.append(new_row)
        new_age_hours.append(new_age_hour_row)
    new_age_hours = np.array(new_age_hours)
        

    # # the number of unique binary features
    num_features = len(mapping)
    print(num_features)

    # # binarize data
    bindata = np.zeros((len(data), num_features)) # initialize a 2D table
    # print(bindata)
    for i, row in enumerate(new_data): # fill in the table
        # print(i, row)
        for x in row: # for each column
            bindata[i][x] = 1
    
    bindata = np.hstack((bindata, new_age_hours))
    return bindata


if __name__ == "__main__":
        # Open text file    
    lines_train = open('income.train.txt.5k').readlines()
    data_train = [line.strip().split(", ") for line in lines_train]

    lines_dev = open('income.dev.txt').readlines()
    data_dev = [line.strip().split(", ") for line in lines_dev]
    

    bin_map = make_binary_mapping(data_train)
    bindata_dev = binarize(data_dev, bin_map)
    bindata_train = binarize(data_train, bin_map)

    
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