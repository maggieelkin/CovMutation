import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow import keras
import tensorflow as tf
import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm


def make_model(n_dim, n_nodes):
    """
    create MLP with 1 hidden layer containing n_nodes

    :param n_dim: number of input features
    :type n_dim: int
    :param n_nodes: number of hidden nodes
    :type n_nodes: int
    :return: MLP model
    :rtype: keras.engine.sequential.Sequential
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_nodes, input_dim=n_dim, activation='sigmoid'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def mlp_model(features_train, labels_train, n_nodes=50, val_split=0.2, epochs=20, batch_size=50, **model_kwargs):
    """  """
    model = make_model(n_dim=features_train.shape[1], n_nodes=n_nodes)
    print(model.summary())
    print('Epochs: {}'.format(epochs))
    print('Batch Size: {}'.format(batch_size))
    print('Validation Split: {}'.format(val_split))
    model.fit(features_train, labels_train, validation_split=val_split, epochs=epochs, batch_size=batch_size)
    return model


def get_metrics(labels_test, predictions, probabilities):
    """
    calculate accuracy, F1-score and AUC

    :param labels_test: label of test data set
    :type labels_test: numpy.ndarray
    :param predictions: predictions (1 or 0) from model
    :type predictions: numpy.ndarray
    :param probabilities: predicted proba from model
    :type probabilities: numpy.ndarray
    :return: tuple of accuracy, balanced accuracy, F1-score and AUC
    :rtype: tuple
    """
    acc = metrics.accuracy_score(labels_test, predictions)
    balanced_acc = metrics.balanced_accuracy_score(labels_test, predictions)
    f1 = metrics.f1_score(labels_test, predictions)
    auc = metrics.roc_auc_score(labels_test, probabilities)
    return acc, balanced_acc, f1, auc


def mlp_predict(model, features_test):
    """

    :param model:
    :type model:
    :param features_test:
    :type features_test:
    :return:
    :rtype:
    """
    proba = model.predict(features_test)
    pred = np.round(proba)
    proba = proba.reshape(-1)
    pred = pred.reshape(-1)
    return pred, proba


def cls_predict(cls, features_test):
    """

    :param cls:
    :type cls:
    :param features_test:
    :type features_test:
    :return:
    :rtype:
    """
    pred = cls.predict(features_test)
    classes = list(cls.classes_)
    p_class = classes.index(1)
    proba = cls.predict_proba(features_test)[:, p_class]
    return pred, proba


def train_cls_model(model_type, features_train, labels_train, **model_kwargs):
    """
    function for sklearn conservation model
    one of: logistic_regression, random_forest, svm

    :param labels_train:
    :type labels_train:
    :param features_train:
    :type features_train:
    :param model_type: type of classifier
    :type model_type: str
    :param model_kwargs: parameters for sklearn model
    :type model_kwargs: dict
    :return: sklearn trained model
    :rtype:
    """
    if model_type == 'logistic_regression':
        cls = LogisticRegression(**model_kwargs)
    elif model_type == 'random_forest':
        cls = RandomForestClassifier(**model_kwargs)
    else:
        cls = SVC(probability=True, **model_kwargs)
    cls.fit(features_train, labels_train)
    return cls


def conservation_model(model_type, features, labels, test_size, **model_kwargs):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42,
                                                                                test_size=test_size, stratify=labels)
    if model_type == 'neural_network':
        model = mlp_model(features_train, labels_train, **model_kwargs)
        pred, proba = mlp_predict(model, features_test)
    else:
        model = train_cls_model(model_type, features_train, labels_train, **model_kwargs)
        pred, proba = cls_predict(model, features_test)
    acc, balanced_accuracy, f1, auc = get_metrics(labels_test, pred, proba)
    results = {'acc': acc,
               'balanced_acc': balanced_accuracy,
               'f1': f1,
               'auc': auc}
    return results


def list_combinations(lst):
    """
    creates different permutations of a list, except an empty list

    :param lst: list to combine
    :type lst: list
    :return: list of combinations
    :rtype: list
    """
    combos = []
    for L in range(1, len(lst) + 1):
        for subset in itertools.combinations(lst, L):
            combos.append(subset)
    return combos


def test_feature_combination(data, features_dict, label, model_types=None, results=None, test_size=.2, **model_kwargs):
    """
    test permutations of feature combinations
    :param model_types: if provided, run specific model type in list, otherwise, run all models
    :type model_types: list
    :param data: position data for conservation model
    :type data: pandas.DataFrame
    :param features_dict: dict of key = feature group name, value = feature name list
    :type features_dict: dict
    :param label: column name in data that corresponds to label
    :type label: str
    :param results: if provided, append model results to results data
    :type results: pandas.DataFrame
    :return: results dataframe
    :rtype: pandas.DataFrame
    """
    if results is not None:
        model_results = results
    else:
        model_results = pd.DataFrame()
    if model_types is not None:
        run_models = model_types
    else:
        run_models = ['neural_network', 'logistic_regression', 'random_forest', 'svm']
    feature_groups = list(features_dict.keys())
    feature_combos = list_combinations(feature_groups)
    index = 0
    for combo in tqdm(feature_combos, desc='Testing Feature Combination'):
        feature_names = []
        for feature_group in combo:
            feature_names.extend(features_dict[feature_group])
        combo_run = {'feature_combo': ", ".join(combo), 'n_features': len(feature_names)}
        model_data = data[[label] + feature_names]
        features = model_data.loc[:, model_data.columns != label].values
        labels = model_data[label].values
        for model_type in run_models:
            combo_run['model'] = model_type
            results = conservation_model(model_type, features, labels, test_size, **model_kwargs)
            combo_run.update(results)
            model_results = model_results.append(pd.DataFrame(combo_run, index=[index]))
            index = index + 1
    return model_results





