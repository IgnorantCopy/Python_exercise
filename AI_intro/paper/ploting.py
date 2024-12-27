import matplotlib.pyplot as plt
import numpy as np
import re


glue_tasks = ["MNLI", "QQP", "QNLI", "SST-2", "CoLA", "MRPC", "RTE"]
glue_models = ["RoBERTa", "RoBERTa-D", "uncased BERT-D", "cased BERT-D", "ALBERT-v1", "ALBERT-v2", "ELECTRA-base", "ELECTRA-small"]
glue_marker = ['o', 'P', 'd', 'v', 'h', '*', '^', 's']
maps = {
    8: ("results/glue", glue_tasks, glue_models, glue_marker),
}


def weight(t1, t2):
    return 1. / (1 + np.exp(-(t1 - t2)))


def sgn(x1, x2):
    if x1 >= x2:
        return 1
    elif x1 < x2:
        return -1


def tau_w(num_of_models: int, T: list, S: list):
    tau = 0.
    for i in range(num_of_models):
        for j in range(i + 1, num_of_models):
            # tau += weight(T[i], T[j]) * sgn(S[i], S[j]) * sgn(T[i], T[j])
            tau += sgn(S[i], S[j]) * sgn(T[i], T[j])
    return (2. / (num_of_models * (num_of_models - 1))) * tau


def read_data(file_name: str):
    T = []
    S = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if line:
                result = re.findall(r"-?\d+\.\d+", line)
                if result:
                    S.append(float(result[0]))
                    T.append(float(result[1]))
    return T, S


def main(num_of_models: int):
    T, S = read_data(maps[num_of_models][0])
    tau_ws = []
    ncols = len(T) // num_of_models
    for i in range(ncols):
        tau_ws.append(tau_w(num_of_models, T[i * num_of_models:(i + 1) * num_of_models], S[i * num_of_models:(i + 1) * num_of_models]))
    plt.subplots(1, ncols, figsize=(20, 2))
    plt.subplots_adjust(hspace=0.3, wspace=0.5)
    for i in range(ncols):
        plt.subplot(1, ncols, i + 1)
        for j in range(num_of_models):
            plt.scatter(T[i * num_of_models + j], S[i * num_of_models + j], label=maps[num_of_models][2][j], marker=maps[num_of_models][3][j], c='black')
        plt.title(f"{maps[num_of_models][1][i]}({tau_ws[i]:.2f})")
    plt.legend(loc=2, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    main(8)
