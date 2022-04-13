def binarize(data, mapping, numerical_feaures=[0,7]):
    num_features = len(mapping) + len(num_features)
    binarized_features = np.zeros([data.shape[0], num_features])
    outcomes = []
    for i, row in enumerate(data):
        for j, x in enumerate(row):
            if j == 0: # Skip the age, hours worked, and maybe the pos/neg outcome
                binarized_features[i, j] = scale_age(int(x))
            elif j == 7:
                binarized_features[i, j] = scale_hours(int(x))
            elif j == 9:
                outcomes.append(1 if x==">50K" else 0)
            else:
                try:
                    binarized_features[i][mapping[j, x]] = 1
                except KeyError:
                    pass
    
    return binarized_features, outcomes