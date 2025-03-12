# %%
# Preliminaries
import sys
sys.path.append('../src')
import mnist_loader
import network2
import network2_L1

# %%
# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# %%
# Train without regularization
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# %%
# Train with L2 regularization
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# %%
# Train with L1 regularization
net = network2_L1.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# %%
# Train with L1 regularization, reduce lambda
net = network2_L1.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 1.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# %%
# Train with L1 regularization, reduce lambda further
net = network2_L1.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 0.2, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
