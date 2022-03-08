# -*- coding: UTF-8 -*-
from sklearn.metrics import matthews_corrcoef  # MCC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings("ignore")


def evaluate(y_true, y_pred, y_score=None):

    cm = confusion_matrix(y_true, y_pred)
    FP = cm[0][1]
    TN = cm[0][0]

    if (FP + TN) != 0:
        spec = float(float(TN) / (float(FP + TN)))
    else:
        raise ValueError('spec = error')

    dic = {}
    dic["acc"] = accuracy_score(y_true, y_pred, normalize=True)
    dic['spec'] = spec
    dic['sen'] = recall_score(y_true, y_pred)
    dic["precision"] = precision_score(y_true, y_pred)
    dic['f1_score'] = f1_score(y_true, y_pred)
    dic['mcc'] = matthews_corrcoef(y_true, y_pred)
    dic["auc"] = roc_auc_score(y_true, y_score)

    return dic


if __name__ == '__main__':
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]
    y_score = [0.49, 0.88, 0.11]
    metric = evaluate(y_true, y_pred, y_score)
    print(metric)
