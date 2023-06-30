import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow import cast, int32


def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)

    return shapes


def set_shape(weights, shapes):
    new_weights = []
    index = 0
    for shape in shapes:
        if len(shape) > 1:
            n_nodes = np.prod(shape) + index
        else:
            n_nodes = shape[0] + index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index = n_nodes

    return new_weights


def get_con_mat(true, pred):
    y_pared_binary = cast(pred > 0.5, dtype=int32).numpy()
    con_1 = confusion_matrix(true, y_pared_binary)
    plt.style.use('seaborn-deep')
    plt.figure(figsize=(5, 5))
    sns.heatmap(con_1, annot=True, annot_kws={'size': 15}, linewidths=0.5, fmt="d", cmap="Blues")
    plt.title('Confusion matrix\n', fontweight='bold', fontsize=15)
    plt.show()


def evaluate_nn(true, pred, train=True):
    if train:
        clf_report = pds.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

    else:
        clf_report = pds.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
