import pandas as pd
import numpy as np

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def replace_missing_values_with_mode(data):
    features = data[['Alley', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'Fence', 'MiscFeature']]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])

    return data

if __name__ == '__main__':
    data = open_file('../resources/train.csv')
    replace_missing_values_with_mode(data)
    print(data)
