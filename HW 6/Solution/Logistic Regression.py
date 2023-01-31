# CSE 404 Introduction to Machine Learning
# Python Lab for Logistic Regression.
# HW 6
# By Sayem Lincoln 
#PID - A54207835

import matplotlib.pyplot as plt
import numpy as np
np.seterr('ignore')

def download(filename): # Returns the data and target values
    with open(filename, 'r') as file:
        filtered = np.array(tuple(map(lambda x: x.rstrip(' ').split(' '),
        filter(lambda x: x[0] == '1' or x[0] == '5', # Filtering
        file.read().rstrip('\n').split('\n')))), np.float32)
        return filtered[:, 1:].reshape((len(filtered), 16, 16)), np.vectorize(lambda x: x == 1)(filtered[:, 0])


def features(data): # Computes the features
    return np.array(( # Average intensity and symmetry
        np.fromiter(( # Average intensity
            0.5 + np.mean(data[i]) / 2 # Converting to range [0, 1]
            for i in range(len(data))), dtype=np.float32, count=len(data)
        ),
        -np.mean(( # Symmetry
            tuple( # Horizontal asymmetry
                sum(np.mean(np.abs(data[i, j] - data[i, -j - 1])) for j in range(8))
                for i in range(len(data))
            ),
            tuple( # Vertical asymmetry
                sum(np.mean(np.abs(data[i, :, j] - data[i, :, -j - 1])) for j in range(8))
                for i in range(len(data)))
            ), axis=0, dtype=np.float32
        )
    )).T


train_data, train_target = download('ZipDigits.train')
test_data, test_target = download('ZipDigits.test')

# Giving a plot of two of the digit images
fig = plt.figure(figsize=(2, 2))
fig.subplots_adjust(left=0.03, right=0.97, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(2):
    ax = fig.add_subplot(1, 2, i + 1, xticks=[], yticks=[])
    ax.imshow(train_data[i], cmap=plt.cm.binary)
plt.show()

# Giving a 2-D scatter plot of features
train_features = features(train_data)
test_features = features(test_data)
features = np.concatenate((train_features, test_features))
target = np.concatenate((train_target, test_target))
plt.scatter(features[target, 0], features[target, 1], marker='1', alpha=0.6)
plt.scatter(features[~target, 0], features[~target, 1], marker='2', alpha=0.6)
plt.legend(['Digit 1', 'Digit 5'])
plt.xlabel('Average intensity')
plt.ylabel('Symmetry')
plt.title('Scatter plot of features')
plt.show()


class LogisticRegression:
    def __init__(self, third_order=False, lr=0.1, num_iter=50000, fit_intercept=True, verbose=False):
        self.third_order=third_order
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def polynomial_transform(self, X):
        for i in range(2, 4):
            for j in range(i + 1):
                X = np.concatenate((X, (np.power(X[:, 0], i - j) * np.power(X[:, 1], j))[:, None]), axis=1)
        return X

    def add_intercept(self, X): # Adding intercept
        if self.third_order:
            return np.concatenate((np.ones((X.shape[0], 1)), self.polynomial_transform(X)), axis=1)
        else:
            return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        h = self.sigmoid(np.dot(X, self.theta))
        return (-np.log(h) * y + np.log(1 - h) * (y - 1)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        self.theta = np.zeros(X.shape[1]) # Weights initialization
        for i in range(self.num_iter):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient # Updating weights

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


model = LogisticRegression()
model.fit(train_features, train_target)
print(
    'Accuracy on train dataset: '
    f'{round(100 * np.mean(model.predict(train_features) == train_target), 2)}%\n'
    f'Ein: {round(100 * model.loss(train_features, train_target), 2)}%\n'
    'Accuracy on test dataset: '
    f'{round(100 * np.mean(model.predict(test_features) == test_target), 2)}%\n'
    f'Etest: {round(100 * model.loss(test_features, test_target), 2)}%\n'
)

plt.scatter(train_features[train_target, 0], train_features[train_target, 1], marker='1', alpha=0.6)
plt.scatter(train_features[~train_target, 0], train_features[~train_target, 1], marker='2', alpha=0.6)
plt.legend(['Digit 1', 'Digit 5'])
plt.xlabel('Average intensity')
plt.ylabel('Symmetry')
plt.title('Training data together with separator')
x1_min, x1_max = train_features[:, 0].min(), train_features[:, 0].max(),
x2_min, x2_max = train_features[:, 1].min(), train_features[:, 1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
plt.show()

plt.scatter(test_features[test_target, 0], test_features[test_target, 1], marker='1', alpha=0.6)
plt.scatter(test_features[~test_target, 0], test_features[~test_target, 1], marker='2', alpha=0.6)
plt.legend(['Digit 1', 'Digit 5'])
plt.xlabel('Average intensity')
plt.ylabel('Symmetry')
plt.title('Test data together with separator')
x1_min, x1_max = test_features[:, 0].min(), test_features[:, 0].max(),
x2_min, x2_max = test_features[:, 1].min(), test_features[:, 1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
plt.show()

model = LogisticRegression(third_order=True)
model.fit(train_features, train_target)
print(
    'Accuracy on train dataset: '
    f'{round(100 * np.mean(model.predict(train_features) == train_target), 2)}%\n'
    f'Ein: {round(100 * model.loss(train_features, train_target), 2)}%\n'
    'Accuracy on test dataset: '
    f'{round(100 * np.mean(model.predict(test_features) == test_target), 2)}%\n'
    f'Etest: {round(100 * model.loss(test_features, test_target), 2)}%\n'
)
