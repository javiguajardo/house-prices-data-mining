from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

import pandas as pd
import numpy

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def select_k_best_features(data, n_atributes):
    # import data
    data = data.fillna(0)
    array = data.values
    num_cols = len(data.columns) - 1
    le = preprocessing.LabelEncoder()
    for i in range(num_cols):
        array[:, i] = le.fit_transform(array[:, i])

    print(array)

    features = array[:, 0:80]
    targets = array[:, 80]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(targets[:10]))

    # Model declaration
    kbest = SelectKBest(score_func = chi2, k = n_atributes)

    # Model training
    kbest.fit(features, targets)

    # Model transformation
    new_feature_vector = kbest.transform(X)

    # Summarize the selection of the attributes
    # Model information:
    print('\nModel information:\n')
    print('Feature Scores: ' + str(kbest.scores_))

    # Model transformation
    new_feature_vector = kbest.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        dict = numpy.unique(numpy_data[:,i])
        # print(dict)
        for j in range(len(dict)):
            # print(numpy.where(numpy_data[:,i] == dict[j]))
            temp[numpy.where(numpy_data[:,i] == dict[j])] = j

        numpy_data[:,i] = temp

    return numpy_data

if __name__ == '__main__':
    data = open_file("../resources/train.csv")
    select_k_best_features(data, 10)
