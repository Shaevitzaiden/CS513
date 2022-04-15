import numpy as np
# import pandas as pd


#################### binarization by pandas ####################
# # set the names of fields, otherwise the first line of the file "toy.txt" would be recognized as a header line
# names = ["age", "sector"]
# f = pd.read_csv("toy.txt", names = names)
# # print(f)
# # print(f["age"])
# # print(type(f["age"]))

# # binarize the field "age"
# # and convert it to numpy array
# bi_age = np.array(pd.get_dummies(f["age"]), dtype=int)
# # print(bi_age)

# # binarize the field "sector"
# bi_sector = np.array(pd.get_dummies(f["sector"]), dtype=int)

# # concatenate two binarized arrays by column-wise
# bi_data = np.concatenate((bi_age, bi_sector), axis=1)
# print(bi_data)


#################### binarization from scratch ####################
# read the file "toy.txt"
lines = open("income.train.txt.5k").readlines()
# print(lines)

# process lines, remove the new line signal "\n" and split each line into two items by comma
data = [line.strip().split(", ") for line in lines]
# print(data)

# an alternative way to process lines
data = list(map(lambda line: line.strip().split(", "), lines))
# print(data)

# binarize the data
mapping = {}
new_data = []
for row in data:
    new_row = []
    for j, x in enumerate(row):
        feature = (j, x) # j is the column index and x is the value
        if feature not in mapping: # new feature
            mapping[feature] = len(mapping) # insert a new feature into the index
        new_row.append(mapping[feature])
    new_data.append(new_row)
# print(mapping)
# print(new_data)

# the number of unique binary features
num_features = len(mapping)
print(num_features)

# binarize data
bindata = np.zeros((len(data), num_features)) # initialize a 2D table
# print(bindata)
for i, row in enumerate(new_data): # fill in the table
    # print(i, row)
    for x in row: # for each column
        bindata[i][x] = 1
print(bindata[0,:])
