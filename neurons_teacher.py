import math
import random
import numpy as np

NUM_NEURONS = 5
NUM_INPUT = 3

# learning rate
V = 0.01

WAGES = [random.uniform(-0.1, 0.1)] * (NUM_NEURONS * NUM_INPUT + NUM_NEURONS)


def calculate_sigmoid_function(x: float) -> float:
    return (1 / (1 + math.exp(-x))) * 10


def calculate_sum(input_row: list, start: int, end: int) -> float:
    return float(np.inner(input_row, WAGES[start:end]))


def calculate_sigmoid_function_der(x: float) -> float:
    return math.exp(x) / ((1 + math.exp(x)) ** 2)


def calculate_neurons_in_hidden_layer(input_row: list[float]) -> tuple[list[float], list[float]]:
    sums = []
    results = []
    for i in range(NUM_NEURONS):
        sums.append(calculate_sum(input_row, len(input_row) * i, len(input_row) * i + len(input_row)))
        results.append(calculate_sigmoid_function(sums[i]))
    return sums, results


def calculate_the_output_neuron(input_row: list[float], sums: list[float], results: list[float]) \
        -> tuple[list[float], list[float]]:
    sums.append(calculate_sum(results, NUM_NEURONS * len(input_row), NUM_NEURONS * len(input_row) + NUM_NEURONS))
    results.append(calculate_sigmoid_function(sums[-1]))
    return sums, results


def go_forward(input_row: list[float]) -> tuple[list[float], list[float]]:
    sums, results = calculate_neurons_in_hidden_layer(input_row)
    sums, results = calculate_the_output_neuron(input_row, sums, results)
    return sums, results


def go_backward(input_row: list[float], sums: list[float], results: list[float], expected_output: float) -> float:
    error = results[-1] - expected_output
    neuron_errors = []
    for i in range(NUM_NEURONS):
        neuron_errors.append(error * WAGES[NUM_NEURONS * len(input_row) + i])

    for i in range(NUM_INPUT * NUM_NEURONS):
        WAGES[i] -= calculate_sigmoid_function_der(sums[i // len(input_row)]) * neuron_errors[i // len(input_row)] * input_row[
            i % len(input_row)] * V
    for i in range(NUM_NEURONS):
        WAGES[NUM_NEURONS * NUM_INPUT + i] -= calculate_sigmoid_function_der(sums[-1]) * error * results[i] * V
    quad_error = error ** 2
    return quad_error


def teach_by_row(input_row: list[float], expected_output: float) -> float:
    sums, results = go_forward(input_row)
    quad_error = go_backward(input_row, sums, results, expected_output)
    return quad_error


def teach_by_matrix(dataset):
    errors = []
    for i in range(len(dataset)):
        errors.append(teach_by_row(dataset[i][0:-1], dataset[i][-1]))

    return np.mean(errors)


def teach_predict_self_matrix(full_matrix: list[list[float]]):
    for i in range(len(full_matrix)):
        teach_predict_self_row(full_matrix[i])


def teach_predict_self_row(full_row: list[float]):
    old_quad_error = 0
    for i in range(100):
        new_quad_error = teach_by_row(full_row[0:-1], full_row[-1])
        predicted = go_forward(full_row[0:-1])[1][-1]
        if abs(new_quad_error - old_quad_error) < 0.1 ** 20:
            print("Predicted in epoch :", i)
            print("Predicted : ", predicted)
            print("Actual : ", full_row[-1], "\n")
            return
        old_quad_error = new_quad_error


def teach_neuron_network(matrix):
    old_error = 10
    for i in range(100):
        new_error = teach_by_matrix(matrix)
        if old_error < new_error:
            raise Exception("Увага! Нейронка розбігається, поставте меншу швидкість навчання V", V)
        if abs(old_error - new_error) < 0.1 ** 5:
            print("Середня квадратична похибка :", new_error)
            print("Завершення циклу через допустиму квадратичну похибку")
            return True
        old_error = new_error


def get_predicted_numbers(input_matrix: list[list[float]]) -> list[float]:
    predicted_list: list[float] = []
    for i in range(len(input_matrix)):
        predicted_list.append(go_forward(input_matrix[i])[1][-1])
    return predicted_list
