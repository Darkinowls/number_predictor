from neurons_teacher import NUM_INPUT, teach_predict_self_row, teach_neuron_network, get_predicted_numbers, np


def init_dataset() -> tuple[list, list]:
    number_row = [0.58, 3.38, 0.91, 5.8,
                  0.91, 5.01, 1.17, 4.67,
                  0.6, 4.81, 0.53, 4.75,
                  1.01, 5.04, 1.07]
    full_matrix: list[list[float]] = []
    input_matrix: list[list[float]] = []
    for i in range(len(number_row) - NUM_INPUT):
        full_matrix.append(number_row[i:i + 4])
        input_matrix.append(number_row[i:i + 3])
    # print(np.matrix(full_matrix))
    # print(np.matrix(input_matrix))
    return full_matrix, input_matrix


if __name__ == '__main__':
    full_matrix, input_matrix = init_dataset()

    for i in range(len(full_matrix)):
        teach_predict_self_row(full_matrix[i])

    for i in range(100):
        print("epoch:", i * 100)
        is_stop = teach_neuron_network(full_matrix[0:-2])
        predicted_list = get_predicted_numbers(input_matrix)
        delta = []
        for j in range(len(input_matrix)):
            delta.append(abs(predicted_list[j] - full_matrix[j][-1]))
            print("Expected :", full_matrix[j][-1], ", predicted :", predicted_list[j], ", delta :", delta[j])
        print("Середня арифметична похибка ", np.mean(delta), "\n")
        if is_stop:
            break
