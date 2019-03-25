from datasets import digits, phoneme
from sklearn import cluster, metrics


DATASET = digits
N_CLUSTERS = 10

report = {}

kmeans = cluster.KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(DATASET.training_features)

for i, c in enumerate(kmeans.labels_):
    if c not in report:
        report[c] = {}
    real_label = DATASET.training_labels[i]
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