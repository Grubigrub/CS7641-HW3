import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from datasets import digits, wave
from sklearn import decomposition, cluster, metrics, neural_network, model_selection
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def vector_distance(x, y):
    d = [y[i] - x[i] for i in range(len(x))]
    s = 0
    for i in range(len(d)):
        s += d[i]*d[i]
    return sqrt(s)

DATASET = digits
N_COMPONENTS = 15
N_CLUSTERS = 10
MODE = 'compute_time'

# General Options
TITLE = 'Neural Network Classifier'
LEARNING_RATE = 1e-2
TOLERANCE = 1e-4
TOPOLOGY = (30,)

ica = decomposition.FastICA(N_COMPONENTS, max_iter=100000, tol=4.8e-4)
new_data = ica.fit_transform(DATASET.training_features)
print(ica.components_)

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
    classifier_ica = neural_network.MLPClassifier(
        learning_rate_init=LEARNING_RATE,
        tol=TOLERANCE,
        hidden_layer_sizes=TOPOLOGY)
    classifier = neural_network.MLPClassifier(
        learning_rate_init=LEARNING_RATE,
        tol=TOLERANCE,
        hidden_layer_sizes=TOPOLOGY)

    plot_x = []
    plot_y_testing = []
    plot_y_mean = []

    train_sizes = np.linspace(0.1, 0.9, 30)

    train_size_abs, train_scores, test_scores = model_selection.learning_curve(
        classifier, DATASET.training_features, DATASET.training_labels,
        cv=10, train_sizes=train_sizes)

    train_size_abs_ica, train_scores_ica, test_scores_ica = model_selection.learning_curve(
        classifier_ica, new_data, DATASET.training_labels,
        cv=10, train_sizes=train_sizes)

    train_losses = [1 - np.array(a).mean() for a in train_scores]
    test_losses = [1 - np.array(a).mean() for a in test_scores]

    train_losses_ica = [1 - np.array(a).mean() for a in train_scores_ica]
    test_losses_ica = [1 - np.array(a).mean() for a in test_scores_ica]

    plt.figure()
    plt.grid()
    plt.xlabel('Training Set Size')
    plt.ylabel('Loss')
    plt.title(TITLE)
    plt.plot(train_size_abs, train_losses)
    plt.plot(train_size_abs, test_losses)
    plt.plot(train_size_abs_ica, train_losses_ica)
    plt.plot(train_size_abs_ica, test_losses_ica)
    plt.legend(['Training original', 'Testing original', 'Training ICA', 'Testing ICA'])
    plt.show()
elif MODE == 'compute_time':
    plot_x = []
    plot_y_scoring = []
    plot_y_fitting = []
    plot_y_scoring_pca = []
    plot_y_fitting_pca = []

    for training_fraction in np.linspace(0.1, 0.9, 10):
        training_size = int(len(new_data) * training_fraction)
        print("Computing score for training_size = {} ...".format(training_size))
        classifier_ica = neural_network.MLPClassifier(
            learning_rate_init=LEARNING_RATE,
            tol=TOLERANCE,
            hidden_layer_sizes=TOPOLOGY)
        classifier = neural_network.MLPClassifier(
            learning_rate_init=LEARNING_RATE,
            tol=TOLERANCE,
            hidden_layer_sizes=TOPOLOGY)

        result = model_selection.cross_validate(
                classifier, 
                DATASET.training_features[:training_size],
                DATASET.training_labels[:training_size],
                cv=10, return_train_score=True)
        result_ica = model_selection.cross_validate(
                classifier_ica, 
                new_data[:training_size],
                DATASET.training_labels[:training_size],
                cv=10, return_train_score=True)
        test_score = result['test_score']
        train_score = result['train_score']
        fit_time = result['fit_time']
        score_time = result['score_time']

        test_score_pca = result_ica['test_score']
        train_score_pca = result_ica['train_score']
        fit_time_pca = result_ica['fit_time']
        score_time_pca = result_ica['score_time']

        plot_x.append(training_size)
        plot_y_scoring.append(np.array(score_time).mean())
        plot_y_fitting.append(np.array(fit_time).mean())

        plot_y_scoring_pca.append(np.array(score_time_pca).mean())
        plot_y_fitting_pca.append(np.array(fit_time_pca).mean())
    
    plt.figure()
    plt.grid()
    plt.xlabel('Set Size')
    plt.ylabel('Compute time in seconds')
    plt.title(TITLE)
    plt.plot(plot_x, plot_y_scoring)
    plt.plot(plot_x, plot_y_fitting)
    plt.plot(plot_x, plot_y_scoring_pca)
    plt.plot(plot_x, plot_y_fitting_pca)
    plt.legend(['Scoring', 'Fitting', 'Scoring ICA', 'Fitting ICA'])
    plt.show()
elif MODE == 'reconstruction':
    inverse_new_data = ica.inverse_transform(new_data)
    max_distance = 0
    max_index = 0
    average_distance = 0
    for i in range(len(DATASET.training_features)):
        original = DATASET.training_features[i]
        reconstructed = inverse_new_data[i]
        d = vector_distance(original, reconstructed)
        average_distance += d
        if d > max_distance:
            max_distance = d
            max_index = i
        average_distance /= len(DATASET.training_features)
    print(max_distance)
    print(average_distance)
