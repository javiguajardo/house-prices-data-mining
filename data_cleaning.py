import pandas as pd
import numpy as np

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def replace_missing_values_with_mode(data):
    features = data[['Alley', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'Fence', 'MiscFeature']]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])

    return data

def replace_missing_values_with_mean(data):
    features = data['LotFrontage']
    mean = features.mean()
    data['LotFrontage'] = data['LotFrontage'].fillna(mean)

    return data

if __name__ == '__main__':
    data = open_file('../resources/train.csv')
    replace_missing_values_with_mode(data)
    replace_missing_values_with_mean(data)
    print(data[:10])
