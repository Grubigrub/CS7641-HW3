import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from datasets import digits, wave
from sklearn import random_projection, cluster, metrics, neural_network, model_selection
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def vector_distance(x, y):
    d = [y[i] - x[i] for i in range(len(x))]
    s = 0
    for i in range(len(d)):
        s += d[i]*d[i]
    return sqrt(s)

DATASET = wave
N_LABELS = 3
N_COMPONENTS = 10
N_CLUSTERS = 3
MODE = 'compute time'

# General Options
TITLE = 'Neural Network Classifier'
N_REPEAT = 10
LEARNING_RATE = 1e-1
TOLERANCE = 1e-4
TOPOLOGY = (3,)

rp = random_projection.GaussianRandomProjection(n_components=N_COMPONENTS)
new_data = rp.fit_transform(DATASET.training_features)

report = {}
labels = []

kmeans = cluster.KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(new_data)

for i, c in enumerate(kmeans.labels_):
    if c not in report:
        report[c] = {}
    real_label = DATASET.training_labels[i]
    if real_label not in labels:
        labels.append(real_label)
    if real_label not in report[c]:
        report[c][real_label] = 1
    else:
        report[c][real_label] += 1

for key in sorted(report):
    value = report[key]
    print("Cluster #{}".format(key))
    max_label_count = 0
    max_label = 0
    for key2 in sorted(value):
        value2 = value[key2]
        if value2 > max_label_count:
            max_label_count = value2
            max_label = key2
        print("\tLabel {}: {} instance(s)".format(key2, value2))
    print("\tMost instances for label {}\n".format(max_label))

if MODE == 'learning':
    test_scores = np.array([])
    train_scores = np.array([])
    test_scores_rp = np.array([])
    train_scores_rp = np.array([])

    for i in range(N_REPEAT):
        if i % 50 == 0:
            print("{}/{}".format(i, N_REPEAT))
        classifier = neural_network.MLPClassifier(
            learning_rate_init=LEARNING_RATE,
            tol=TOLERANCE,
            hidden_layer_sizes=TOPOLOGY)
        classifier_rp = neural_network.MLPClassifier(
            learning_rate_init=LEARNING_RATE,
            tol=TOLERANCE,
            hidden_layer_sizes=TOPOLOGY)

        _, train, test = model_selection.learning_curve(
            classifier, DATASET.training_features, DATASET.training_labels,
            cv=10, train_sizes=[1.0])

        _, train_rp, test_rp = model_selection.learning_curve(
            classifier_rp, new_data, DATASET.training_labels,
            cv=10, train_sizes=[1.0])
        
        test_scores = np.append(test_scores, [test])
        train_scores = np.append(train_scores, [train])

        test_scores_rp = np.append(test_scores_rp, [test_rp])
        train_scores_rp = np.append(train_scores_rp, [train_rp])

    print("Original: Test: {}    Train: {}".format(test_scores.mean(), train_scores.mean()))
    print("RP:       Test: {}    Train: {}".format(test_scores_rp.mean(), train_scores_rp.mean()))

if MODE == 'compare':
    success = 0
    for i in range(1000):
        if i % 50 == 0:
            print("{}/{}".format(i, 1000))
        rp = random_projection.GaussianRandomProjection(n_components=N_COMPONENTS)
        new_data = rp.fit_transform(DATASET.training_features)
        kmeans = cluster.KMeans(n_clusters=N_CLUSTERS)
        kmeans.fit(new_data)
        count = {}
        for i, c in enumerate(kmeans.labels_):
            if c not in count:
                count[c] = {}
            if DATASET.training_labels[i] not in count[c]:
                count[c][DATASET.training_labels[i]] = 1
            else:
                count[c][DATASET.training_labels[i]] += 1
        
        maxima = []
        for key, value in count.items():
            maxima.append(max(value, key=value.get))
        maxima = set(maxima)
        if len(maxima) == N_LABELS:
            success += 1
    print(success)
