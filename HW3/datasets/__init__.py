import os
import csv
from random import shuffle, seed
from sklearn import datasets

# Datasets Utility Classes
print(os.getcwd())

class DigitsData():
    data = []
    training = []
    testing = []
    n_testing_samples = 0
    n_training_samples = 0

    def __getattribute__(self, item):
        if item == 'training_features':
            return [a[1] for a in self.training]
        elif item == 'testing_features':
            return [a[1] for a in self.testing]
        elif item == 'training_labels':
            return [a[0] for a in self.training]
        elif item == 'testing_labels':
            return [a[0] for a in self.testing]
        else:
            return super().__getattribute__(item)

    def populate(self):
            n = len(self.data)
            seed(2)
            shuffle(self.data)
            self.training = self.data[n // 7:]
            self.testing = self.data[:n // 7]

    def shuffle(self):
        shuffle(self.training)
        self.n_features = len(self.training[0][1])
	
class WaveformData:
    data = []
    training = []
    testing = []
    n_testing_samples = 0
    n_training_samples = 0

    def __getattribute__(self, item):
        if item == 'training_features':
            return [a[1] for a in self.training]
        elif item == 'testing_features':
            return [a[1] for a in self.testing]
        elif item == 'training_labels':
            return [a[0] for a in self.training]
        elif item == 'testing_labels':
            return [a[0] for a in self.testing]
        else:
            return super().__getattribute__(item)

    def populate(self):
            n = len(self.data)
            shuffle(self.data)
            self.training = self.data[n // 7:]
            self.testing = self.data[:n // 7]
            self.n_features = len(self.training[0][1])

    def shuffle(self):
        shuffle(self.training)



# Digits loading
digits = DigitsData()

input_digits = datasets.load_digits()

for i in range(len(input_digits.target)):
    label = input_digits.target[i]
    features = input_digits.data[i]
    digits.data.append((label, features))

digits.populate()
digits.n_training_samples = len(digits.training)
digits.n_testing_samples = len(digits.testing)

# Waveform Loading
wave = WaveformData()

with open(os.path.join(os.getcwd(), '..', 'datasets/waveform.data'), encoding='utf8') as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    for row in reader:
        label = row[len(row) - 1]
        features = [float(a) for a in row[0:len(row) - 1]]
        wave.data.append((label, features))

wave.populate()
wave.n_training_samples = len(wave.training)
wave.n_testing_samples = len(wave.testing)




