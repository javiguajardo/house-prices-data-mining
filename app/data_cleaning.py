from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

def open_file(file_name):
    data = pd.read_csv(file_name)
    return data

def write_file(data, fileName):
    new_data = pd.DataFrame(data=data).to_csv(fileName)

def replace_missing_values_with_mode(data, features):
    features = data[features]
    columns = features.columns
    mode = data[columns].mode()
    data[columns] = data[columns].fillna(mode.iloc[0])
    return data

def replace_missing_values_with_mean(data, features):
    features = data[features]
    columns = features.columns
    mean = data[columns].mean()
    mean = round(mean, 2)
    data[columns] = data[columns].fillna(mean.iloc[0])

    return data

def replace_missing_values_with_constant(data):
    data['Alley'] = data['Alley'].fillna('NOACCESS')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('NoBSMT')

    data['FireplaceQu'] = data['FireplaceQu'].fillna('NoFP')

    for col in ('GarageType', 'GarageFinish', 'GarageQual'):
        data[col] = data[col].fillna('NoGRG')

    data['Fence'] = data['Fence'].fillna('NOFENCE')
    data['MiscFeature'] = data['MiscFeature'].fillna('NOMISC')

    return data

def replace_outliers(data):
    data.loc[data['Utilities'] == 'NoSeWa', 'Utilities'] = 'AllPub'
    data.loc[data['HeatingQC'] == 'Po', 'HeatingQC'] = data['HeatingQC'].mode()
    return data

def drop_useless_features(data):
    data.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)
    return data

def principal_components_analysis(data, n_components):
    # import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(1, num_features))]
    target = data[[num_features]]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(features)

    # Model transformation
    new_feature_vector = pca.transform(features)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    print('\n\n')

    return new_feature_vector

def attribute_subset_selection_with_trees(data):
    # import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(1, num_features))]
    target = data[[num_features]]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Model declaration
    extra_tree = ExtraTreesClassifier()

    # Model training
    extra_tree.fit(features, target.values.ravel())

    # Model information:
    print('\nModel information:\n')

    # display the relative importance of each attribute
    importances = extra_tree.feature_importances_
    std = np.std([tree.feature_importances_ for tree in extra_tree.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(features.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, features.columns[indices[f]], importances[indices[f]]))

    # If model was training before prefit = True
    model = SelectFromModel(extra_tree, prefit = True)

    # Model transformation
    new_feature_vector = model.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(features.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(features.shape[1]), indices)
    plt.xlim([-1, features.shape[1]])
    plt.show()

def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        dict = pd.unique(numpy_data[:,i])
        # print(dict)
        for j in range(len(dict)):
            # print(numpy.where(numpy_data[:,i] == dict[j]))
            temp[np.where(numpy_data[:,i] == dict[j])] = j

        numpy_data[:,i] = temp

    return numpy_data

def z_score_normalization(data):
    """# import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(1, num_features))]
    target = data[[num_features]]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))"""

    # Data standarization
    standardized_data = preprocessing.scale(data)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])
    print('\n\n')

    return standardized_data

def min_max_scaler(data):
    # import data
    num_features = len(data.columns) - 1

    cols = data.columns
    num_cols = data._get_numeric_data().columns
    nominal_cols = list(set(cols) - set(num_cols))

    data[nominal_cols] = convert_data_to_numeric(data[nominal_cols])

    features = data[list(range(1, num_features))]
    target = data[[num_features]]

    # First 10 rows
    print('Training Data:\n\n' + str(features[:10]))
    print('\n')
    print('Targets:\n\n' + str(target[:10]))

    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(features)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(features)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])

def remove_rows(data, column):
    data = data.dropna(subset=[column], inplace=True)
    return data

def remove_columns(data, column):
    data = data.drop(column, axis=1, inplace=True)
    return data

def replace_sf_columns(data):
    data.insert(len(data.columns) - 2, 'TotalSF', data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'])
    data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
    return data

if __name__ == '__main__':
    data = open_file('../resources/training_data.csv')
    replace_outliers(data)
    remove_rows(data, 'GarageYrBlt')
    #remove_columns(data, 'PoolQC')
    replace_sf_columns(data)
    replace_missing_values_with_mode(data, ['MasVnrType', 'Electrical', 'GarageCond', 'HeatingQC'])
    replace_missing_values_with_mean(data, ['LotFrontage', 'MasVnrArea'])
    replace_missing_values_with_constant(data)
    drop_useless_features(data)
    data = principal_components_analysis(data, 30)
    data = z_score_normalization(data)
    #attribute_subset_selection_with_trees(data)
    #min_max_scaler(data)
    write_file(data, '../resources/output.csv')
