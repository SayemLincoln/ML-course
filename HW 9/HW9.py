import numpy as np
import scipy.io
from svmutil import *
import matplotlib.pyplot as plt

data = scipy.io.loadmat('data.mat')
usps = scipy.io.loadmat('USPS.mat')

X_data = data['X']
Y_data = data['Y']

X_usps = usps['A']
Y_usps = usps['L']

Y_data_new = np.concatenate(Y_data).tolist()
X_data_new = []
for i in range(X_data.shape[0]):
    x_element = X_data[i]
    dictOfX = { i+1 : x_element[i] for i in range(0, len(x_element) ) }
    X_data_new.append(dictOfX)

Y_usps_new = np.concatenate(Y_usps).tolist()
X_usps_new = []
for i in range(X_usps.shape[0]):
    x_element = X_usps[i]
    dictOfX = { i+1 : x_element[i] for i in range(0, len(x_element) ) }
    X_usps_new.append(dictOfX)

### DATA.MAT with varying c
c_value = [0.01, 0.1, 1, 2, 3, 5]

accuracy = []
support_vectors = []
for i in range(len(c_value)):
    model = svm_train(Y_data_new[:150], X_data_new[:150], '-c ' + str(c_value[i]));
    p_label, p_acc, p_val = svm_predict(Y_data_new[150:], X_data_new[150:], model);
    accuracy.append(p_acc[0]);
    support_vectors.append(model.get_nr_sv());

print(accuracy)
print(support_vectors)

fig, axs = plt.subplots(4,2)
axs[0,0].plot(['0.01', '0.1', '1', '2', '3', '5'], accuracy)
axs[0,0].set_title('Accuracy of data.mat with varying c')

axs[0,1].plot(['0.01', '0.1', '1', '2', '3', '5'], support_vectors)
axs[0,1].set_title('Support Vectors of data.mat with varying c')

### DATA.MAT with varying kernels
kernel = [0, 1, 2, 3]

accuracy = []
support_vectors = []
for i in range(len(kernel)):
    model = svm_train(Y_data_new[:150], X_data_new[:150], '-c 1 -t ' + str(kernel[i]));
    p_label, p_acc, p_val = svm_predict(Y_data_new[150:], X_data_new[150:], model);
    accuracy.append(p_acc[0]);
    support_vectors.append(model.get_nr_sv());

print(accuracy)
print(support_vectors)

axs[1,0].plot(['Linear', 'Polynomial', 'RBF', 'Sigmoid'], accuracy)
axs[1,0].set_title('Accuracy of data.mat with varying kernels')

axs[1,1].plot(['Linear', 'Polynomial', 'RBF', 'Sigmoid'], support_vectors)
axs[1,1].set_title('Support Vectors of data.mat with varying kernels')

### USPS.MAT with varying c
c_value = [0.01, 0.1, 1, 2, 3, 5]

accuracy = []
support_vectors = []
for i in range(len(c_value)):
    model = svm_train(Y_usps_new[:2500], X_usps_new[:2500], '-c ' + str(c_value[i]));
    p_label, p_acc, p_val = svm_predict(Y_usps_new[2500:], X_usps_new[2500:], model);
    accuracy.append(p_acc[0]);
    support_vectors.append(model.get_nr_sv());

print(accuracy)
print(support_vectors)

axs[2,0].plot(['0.01', '0.1', '1', '2', '3', '5'], accuracy)
axs[2,0].set_title('Accuracy of usps.mat with varying c')

axs[2,1].plot(['0.01', '0.1', '1', '2', '3', '5'], support_vectors)
axs[2,1].set_title('Support Vectors of usps.mat with varying c')

### USPS.MAT with varying kernels
kernel = [0, 1, 2, 3]

accuracy = []
support_vectors = []
for i in range(len(kernel)):
    model = svm_train(Y_usps_new[:2500], X_usps_new[:2500], '-c 1 -t ' + str(kernel[i]));
    p_label, p_acc, p_val = svm_predict(Y_usps_new[2500:], X_usps_new[2500:], model);
    accuracy.append(p_acc[0]);
    support_vectors.append(model.get_nr_sv());

print(accuracy)
print(support_vectors)

axs[3,0].plot(['Linear', 'Polynomial', 'RBF', 'Sigmoid'], accuracy)
axs[3,0].set_title('Accuracy of usps.mat with varying kernels')

axs[3,1].plot(['Linear', 'Polynomial', 'RBF', 'Sigmoid'], support_vectors)
axs[3,1].set_title('Support Vectors of data.mat with varying kernels')

plt.show()
