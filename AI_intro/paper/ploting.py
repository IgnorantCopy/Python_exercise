import matplotlib.pyplot as plt
import numpy as np
import re


glue_results = {
    "MNLI": {
        "RoBERTa": (-0.5340856709158951, 0.8781456953642384),
        "uncased BERT-D": (-0.2205592725388641, 0.8211920529801324),
        "ALBERT-v2": (-0.5617629793059487, 0.8448293428425879),
        "ELECTRA-base": (-0.1270556164024973, 0.8874172185430463),
        "ELECTRA-small": (-0.4013931383553602, 0.8119205298013245)
    },
    "QQP": {
        "RoBERTa": (-0.5614161174849488, 0.9159040316596587),
        "RoBERTa-D": (-0.738141810989102, 0.6318327974276527),
        "uncased BERT-D": (-0.9129112756344686, 0.5338609943111551),
        "cased BERT-D": (-0.5754286202988488, 0.8974771209497897),
        "ALBERT-v2": (-0.6883012127418151, 0.9049715557754143),
        "ELECTRA-base": (-0.41978923276613933, 0.9193420727182785),
        "ELECTRA-small": (-0.6306593387013033, 0.8947069008162256)
    },
    "QNLI": {
        "RoBERTa": (-0.5525178138247662, 0.9267801574226615),
        "RoBERTa-D": (-0.8472257826613385, 0.4946000366099213),
        "uncased BERT-D": (-0.4704686393263577, 0.8848617975471352),
        "ALBERT-v2": (-0.7711817913454362, 0.9136005857587406),
        "ELECTRA-base": (-0.8251125130076153, 0.4946000366099213)
    },
    "SST-2": {
        "RoBERTa": (-0.7155994137768582, 0.9403669724770642),
        "RoBERTa-D": (-0.8194152449059582, 0.9277522935779816),
        "uncased BERT-D": (-0.2630222378166057, 0.9105504587155964),
        "cased BERT-D": (-0.8326835243171337, 0.9002293577981652),
        "ALBERT-v2": (-0.6331057368447117, 0.9231651376146789),
        "ELECTRA-base": (-0.5110292330649795, 0.9506880733944955),
        "ELECTRA-small": (-0.6847325826760177, 0.9174311926605505)
    },
    "CoLA": {
        "RoBERTa": (-0.40225478819813565, 0.6382594026155579),
        "RoBERTa-D": (-0.4375926363772261, 0.5788207437251082),
        "uncased BERT-D": (-0.792119084211421, 0.5685664296893979),
        "cased BERT-D": (-0.5758762985856327, 0.46372927911071965),
        "ALBERT-v2": (-0.5233847261808, 0.5494768667363472),
        "ELECTRA-base": (-0.4206984419701534, 0.6579677841732349)
    },
    "MRPC": {
        "RoBERTa": (-0.3543709410468441, 0.8970588235294118),
        "RoBERTa-D": (-0.6229590671981178, 0.8308823529411765),
        "uncased BERT-D":(-0.5298636548772029, 0.8578431372549019),
        "cased BERT-D": (-0.7982118975095636, 0.7843137254901961),
        "ALBERT-v2": (-0.7048722692879975, 0.8627450980392157),
        "ELECTRA-base": (-0.549591245533545, 0.8897058823529411),
        "ELECTRA-small": (-0.6972814945905685, 0.8529411764705882)
    },
    "RTE": {
        "RoBERTa": (-0.6421491451004212, 0.7833935018050542),
        "uncased BERT-D": (-0.833935266212873, 0.5992779783393501),
        "ALBERT-v2": (-1.0153694027610904, 0.6750902527075813),
        "ELECTRA-base": (-0.7623099192650991, 0.8231046931407943)
    }
}
glue_tasks = ["MNLI", "QQP", "QNLI", "SST-2", "CoLA", "MRPC", "RTE"]
glue_models = ["RoBERTa", "RoBERTa-D", "uncased BERT-D", "cased BERT-D", "ALBERT-v2", "ELECTRA-base", "ELECTRA-small"]
glue_marker = ['o', 'P', 'd', 'v', '*', '^', 's']
maps = {
    8: (glue_results, glue_tasks, glue_models, glue_marker),
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


def main(num_of_models: int):
    tau_ws = []
    ncols = len(maps[num_of_models][1])
    T = {}
    S = {}
    for k, v in maps[num_of_models][0].items():
        temp_t = []
        temp_s = []
        for model_name, (s, t) in v.items():
            temp_t.append(t)
            temp_s.append(s)
        T[k] = temp_t
        S[k] = temp_s
    for i in range(ncols):
        task_name = maps[num_of_models][1][i]
        tau_ws.append(tau_w(len(T[task_name]), T[task_name], S[task_name]))
    fig = plt.figure(figsize=(25, 3))
    axes = fig.subplots(nrows=1, ncols=ncols)
    # plt.subplots_adjust(hspace=0.3, wspace=0.5)
    for i, ax in enumerate(axes):
        for model_name, (s, t) in maps[num_of_models][0][maps[num_of_models][1][i]].items():
            ax.scatter(t, s, label=model_name, marker=maps[num_of_models][3][maps[num_of_models][2].index(model_name)], c='black')
        ax.set_title(f"{maps[num_of_models][1][i]}({tau_ws[i]:.2f})")
    dots = set()
    labels = set()
    for i in range(ncols):
        dot, label = fig.axes[i].get_legend_handles_labels()
        for x, y in zip(dot, label):
            dots.add(x)
            labels.add(y)
    fig.legend(dots, labels, loc='right')
    # plt.legend(loc=2, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    main(8)
