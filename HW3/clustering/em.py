import os
import sys

sys.path.append(os.path.join(os.getcwd(), os.pardir))

from datasets import digits, wave
from sklearn import mixture, metrics

DATASET = wave
N_CLUSTERS = 3

report = {}

em = mixture.GaussianMixture(N_CLUSTERS)
labels = em.fit_predict(DATASET.training_features)

for i, label in enumerate(labels):
    if label not in report:
        report[label] = {}
    real_label = DATASET.training_labels[i]
    if real_label not in report[label]:
        report[label][real_label] = 1
    else:
        report[label][real_label] += 1

for i, key in enumerate(sorted(report)):
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
