import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Your code here

    probabilities = []

    for sample in features:

        z = bias
        for i in range(len(sample)):
            z += sample[i] * weights[i]

        pred = 1 / (1 + math.exp(-z))
        probabilities.append(round(pred, 4))

    squared_errors = []

    for i in range(len(probabilities)):
        error = (probabilities[i] - labels[i]) ** 2
        squared_errors.append(error)

    mse = round(sum(squared_errors) / len(labels), 4)

    return probabilities, mse