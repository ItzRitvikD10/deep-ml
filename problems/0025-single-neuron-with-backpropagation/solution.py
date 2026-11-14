import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    
    weights = np.array(initial_weights, dtype=float)
    bias = np.array(initial_bias, dtype=float)

    mse_values = []

    n_samples = len(labels)

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = (1 / (1 + np.exp(-z)))

        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(float(mse), 4))

        d_loss_d_z = (2 / n_samples) * errors * predictions * (1 - predictions)

        weights_gradient = np.dot(features.T, d_loss_d_z)
        bias_gradient = np.sum(d_loss_d_z)

        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient

    return np.round(weights, 4), np.round(bias, 4), mse_values