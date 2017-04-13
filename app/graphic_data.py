import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def describe_columns(data):
    describe = data.describe(include="all")
    print(describe)

def calc_var(data):
    var = data.var(skipna=False)
    print(var)

def count_nans_in_row(data, column):
    columns = data.columns
    null = data[column].index[pd.isnull(data[column])]
    not_null = data[column].index[pd.notnull(data[column])]

    subset = data.ix[null]
    subset_not_null = data.ix[not_null]

    null_subset = subset.isnull().sum(axis=1)
    not_null_data = subset_not_null.isnull().sum(axis=1)

    print(null_subset[:10])
    print(not_null_data[:10])


if __name__ == '__main__':
    data = open_file('../resources/train.csv')
    describe_columns(data)
    #calc_var(data)
    #count_nans_in_row(data, 'GarageYrBlt')
