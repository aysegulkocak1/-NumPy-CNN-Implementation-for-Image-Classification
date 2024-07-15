import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, learning_rate, input_neurons, output_neurons, epochs, x_train, x_test, y_train, y_test):
        self.learning_rate = learning_rate
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.epochs = epochs
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.filter_num = 3
        self.filter_height = 3
        self.filter_width = 3
        self.filters = np.random.uniform(size=(self.filter_num, self.filter_height, self.filter_width))
        self.output_height = self.x_train.shape[1] - self.filter_height + 1
        self.output_width = self.x_train.shape[2] - self.filter_width + 1
        self.weights = np.random.uniform(size=(self.output_height * self.output_width * self.filter_num, self.output_neurons))
        self.biases = np.zeros(self.output_neurons)


    def generate_patches(self, data):
        num_data, height, width = data.shape
        new_shape = (num_data, height - self.filter_height + 1, width - self.filter_width + 1, self.filter_height, self.filter_width)
        new_strides = data.strides + data.strides[-2:]
        patches = as_strided(data, shape=new_shape, strides=new_strides)
        patches = patches.reshape(data.shape[0], -1, self.filter_height * self.filter_width)
        return patches


    def relu(self, x):
        return np.maximum(0, x)


    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)


    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def convolve(self, data):
        f = self.filters
        num_data, height, width = data.shape
        output_height = height - self.filter_height + 1
        output_width = width - self.filter_width + 1
        patches = self.generate_patches(data)
        output = np.matmul(patches, f.reshape(self.filter_num, -1).T).reshape(num_data, self.filter_num, output_height, output_width)
        return output


    def forward_propagation(self):
        conv_output = self.convolve(self.x_train)
        relu_output = self.relu(conv_output)
        flat_output = relu_output.reshape(relu_output.shape[0], -1)
        z = np.dot(flat_output, self.weights) + self.biases
        return z, flat_output


    def cross_entropy_loss(self, predicted, actual):
        num_samples = len(predicted)
        loss = -np.sum(actual * np.log(predicted + 1e-10)) / num_samples
        return loss


    def backward_propagation(self, predicted, flat_output):
        num_samples = len(predicted)
        grad_z = predicted - self.y_train
        grad_weights = np.dot(flat_output.T, grad_z) / num_samples
        grad_biases = np.sum(grad_z, axis=0) / num_samples

        conv_output = self.convolve(self.x_train)
        grad_filters = np.zeros_like(self.filters)
        patches = self.generate_patches(self.x_train)

        for i in range(self.filter_num):
            for j in range(self.output_height):
                for k in range(self.output_width):
                    grad_filters[i] += np.sum(
                        grad_z[:, i].reshape(-1, 1, 1) * patches[:, j*self.output_width + k].reshape(-1, self.filter_height, self.filter_width),
                        axis=0
                    )
            grad_filters[i] /= num_samples

        return grad_weights, grad_biases, grad_filters


    def update_parameters(self, grad_weights, grad_biases, grad_filters):
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        self.filters -= self.learning_rate * grad_filters


    def train(self):
        loss_train = []
        accuracy_train = []
        loss_test = []
        accuracy_test = []

        for epoch in range(self.epochs):
            z, flat_output = self.forward_propagation()
            output_softmax = self.softmax(z)
            loss = self.cross_entropy_loss(output_softmax, self.y_train)
            loss_train.append(loss)
            
            predicted_classes = np.argmax(output_softmax, axis=1)
            actual_classes = np.argmax(self.y_train, axis=1)
            accuracy = np.mean(predicted_classes == actual_classes)
            accuracy_train.append(accuracy)
            
            grad_weights, grad_biases, grad_filters = self.backward_propagation(output_softmax, flat_output)
            self.update_parameters(grad_weights, grad_biases, grad_filters)
            
            test_accuracy, test_loss = self.test()
            loss_test.append(test_loss)
            accuracy_test.append(test_accuracy)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%")

        return loss_train, accuracy_train, loss_test, accuracy_test


    def predict(self, x, return_flat_output=False):
        conv_output = self.convolve(x)
        relu_output = self.relu(conv_output)
        flat_output = relu_output.reshape(relu_output.shape[0], -1)
        z = np.dot(flat_output, self.weights) + self.biases
        output_softmax = self.softmax(z)
        if return_flat_output:
            return z, flat_output
        return output_softmax


    def test(self):
        z, flat_output = self.predict(self.x_test, return_flat_output=True)
        output_softmax = self.softmax(z)
        
        loss = self.cross_entropy_loss(output_softmax, self.y_test)
        
        predicted_classes = np.argmax(output_softmax, axis=1)
        actual_classes = np.argmax(self.y_test, axis=1)
        accuracy = np.mean(predicted_classes == actual_classes)

        return accuracy, loss



    def plot_graphics(self, loss_train, loss_test, accuracy_train, accuracy_test):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(loss_train, label='Train Loss')
        plt.plot(loss_test, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracy_train, label='Train Accuracy')
        plt.plot(accuracy_test, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode labels
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y]

x_train, x_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.33, random_state=42)

x_train = x_train.reshape(-1, 8, 8)
x_test = x_test.reshape(-1, 8, 8)

cnn = CNN(learning_rate=0.001, input_neurons=X.shape[1], output_neurons=num_classes, epochs=5000,
          x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

loss_train, accuracy_train, loss_test, accuracy_test = cnn.train()
cnn.test()
cnn.plot_graphics(loss_train, loss_test, accuracy_train, accuracy_test)
