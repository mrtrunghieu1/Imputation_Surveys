# Library
import sys
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from impyute.imputation.cs import mice
import numpy as np


# My Library

def imputation_fit(missing_data_train, flag):
    if flag == 0:
        imp = SimpleImputer(strategy="mean")
        imp.fit(missing_data_train)
        imp_name = SimpleImputer.__name__ + "_" + imp.strategy
    elif flag == 1:
        imp = SimpleImputer(strategy="median")
        imp.fit(missing_data_train)
        imp_name = SimpleImputer.__name__ + "_" + imp.strategy
    elif flag == 2:
        imp = SimpleImputer(strategy="most_frequent")
        imp.fit(missing_data_train)
        imp_name = SimpleImputer.__name__ + "_" + imp.strategy
    elif flag == 3:
        imp = SimpleImputer(strategy="constant")
        imp.fit(missing_data_train)
        imp_name = SimpleImputer.__name__ + "_" + imp.strategy
    elif flag == 4:
        x_train = missing_data_train[:, :(missing_data_train.shape[1] - 1)]
        y_train = missing_data_train[:, -1]
        imp = KNNImputer(n_neighbors=5)
        imp.fit(x_train, y_train)
        imp_name = imp.__class__.__name__
    elif flag == 5:
        imp = None
        imp_name = "MICE"
    else:
        raise Exception("Not methods imputation!!!")
    return imp, imp_name


def imputation_transform(model, missing_data_train, missing_test, flag, missingness):
    if flag == 4:
        x_train = missing_data_train[:, :(missing_data_train.shape[1] - 1)]
        y_train = missing_data_train[:, -1]
        x_test = missing_test[:, :(missing_test.shape[1] - 1)]
        y_test = missing_test[:, -1]
        X_train_predicted = model.transform(x_train)
        X_test_predicted = model.transform(x_test)
        y_tr_reshape = y_train.reshape(y_train.shape[0], 1)
        y_te_reshape = y_test.reshape(y_test.shape[0], 1)
        train_predict = np.concatenate((X_train_predicted, y_tr_reshape), axis=1)
        test_predict = np.concatenate((X_test_predicted, y_te_reshape), axis=1)
    elif flag == 5:
        if missingness != 0:
            train_predict = mice(missing_data_train)
            test_predict = mice(missing_test)
        else:
            train_predict = missing_data_train
            test_predict = missing_test
    else:
        train_predict = model.transform(missing_data_train)
        test_predict = model.transform(missing_test)
    return train_predict, test_predict


def imputation_method(missing_data_train, missing_test, flag, missingness):
    model, imp_name = imputation_fit(missing_data_train, flag)
    train_predict, test_predict = imputation_transform(model, missing_data_train, missing_test, flag, missingness)
    return train_predict, test_predict, imp_name
