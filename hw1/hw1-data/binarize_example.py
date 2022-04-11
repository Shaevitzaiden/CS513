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


if __name__ == "__main__":
    # Open text file    
    lines = open("income.train.txt.5k").readlines()

    # process lines, remove the new line signal "\n" and split each line into two items by comma
    data = [line.strip().split(", ") for line in lines]
    # print(data)

    # create binarization mapping where "new_data" is lists of integers corresponding to hot indices in the binarized matrix
    mapping = {}
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
                if feature not in mapping: # new feature
                    mapping[feature] = len(mapping) # insert a new feature into the index
                new_row.append(mapping[feature])
        new_data.append(new_row)
        new_age_hours.append(new_age_hour_row)
    new_age_hours = np.array(new_age_hours)
    # print(mapping)
    # print(new_data)

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
    
