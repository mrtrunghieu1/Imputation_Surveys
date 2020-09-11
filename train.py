# Library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import tree


# My Library

# Training phrase
def training_model(data_train, label_train, flag, n_classes):
    if flag == 0:
        kneigh = KNeighborsClassifier(n_neighbors=10)
        name_classification_algo = kneigh.__class__.__name__
        model = kneigh.fit(data_train, label_train)
    elif flag == 1:
        rf = RandomForestClassifier(n_estimators=200)
        name_classification_algo = rf.__class__.__name__
        model = rf.fit(data_train, label_train)
    elif flag == 2:
        if n_classes == 2:
            model = XGBClassifier(n_estimators=200, object='binary:logistic')
        else:
            model = XGBClassifier(n_estimators=200, object='multi:softmax')
        name_classification_algo = model.__class__.__name__
        model.fit(data_train, label_train)
    elif flag == 3:
        clf = tree.DecisionTreeClassifier()
        name_classification_algo = clf.__class__.__name__
        model = clf.fit(data_train, label_train)
    return model, name_classification_algo


# Testing phrase
def testing_model(data_test, model):
    label_predict = model.predict(data_test)
    return label_predict


def model_prediction(data_train, data_test, label_train, flag, n_classes):
    model, name_classification_algo = training_model(data_train, label_train, flag, n_classes)
    label_predicted = testing_model(data_test, model)
    return label_predicted, name_classification_algo
