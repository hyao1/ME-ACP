# -*- coding: UTF-8 -*-
import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from dataset import ACPDataset
from train import train
from lightgbm.sklearn import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def file_reading(filepath):
    dataset2 = pd.read_csv(filepath[0], header=None, low_memory=False, dtype=np.float32)
    dataset4 = pd.read_csv(filepath[1], header=None, low_memory=False, dtype=np.float32)

    # only be used except for lightGBM
    # dataset2 = dataset2.fillna(0)
    # dataset4 = dataset4.fillna(0)

    train_data = pd.concat([dataset2, dataset4], axis=0)
    neg_train_tags = [0] * dataset2.shape[0]
    pos_train_tags = [1] * dataset4.shape[0]
    train_tags = neg_train_tags + pos_train_tags

    data = [train_data, train_tags]
    return data


def generate_data(filegroup, method):
    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    method_data = {}

    for methodname in method:
        for i in postrain:
            if methodname in i:
                postrain_method = i
                break
        for j in negtrain:
            if methodname in j:
                negtrain_method = j
                break

        filepath = [negtrain_method, postrain_method]

        method_data[methodname] = file_reading(filepath)
    return method_data


def train_ML_model(datadic):
    train_feature = {}
    test_feature = {}

    for i in datadic:
        data = datadic[i]
        y_pred_train, y_pred_test = machine_learning_train(data[0], data[1], data[2])
        train_feature[i] = y_pred_train
        test_feature[i] = y_pred_test

    train_feature_vector = pd.DataFrame(train_feature)
    test_feature_vector = pd.DataFrame(test_feature)
    data[0] = train_feature_vector.values
    data[2] = test_feature_vector.values
    return data


def machine_learning_train(traindata, traintags, testdata):
    clf = LGBMClassifier()

    clf.fit(traindata, traintags)
    train_label = clf.predict_proba(traindata)
    y_score = clf.predict_proba(testdata)

    return train_label[:, 1], y_score[:, 1]


def get_feature(arg):
    if arg.data_name == 'ACP240':
        tmpdir = "Anti-cancer-data/ACP-740-240-feature"
        postrain = glob.glob(tmpdir + '/ACP-240-feature/ACP240_positive*')
        negtrain = glob.glob(tmpdir + '/ACP-240-feature/ACP240_negative*')
        data_len = 240
        min_len = 11
        seed = 1

    elif arg.data_name == 'ACP740':
        tmpdir = "Anti-cancer-data/ACP-740-240-feature"
        postrain = glob.glob(tmpdir + '/ACP-740-feature/ACP740_positive*')
        negtrain = glob.glob(tmpdir + '/ACP-740-feature/ACP740_negative*')
        data_len = 740
        min_len = 11
        seed = 5
    else:
        raise ValueError('No dataset name!')

    filegroup = {}
    filegroup['postrain'] = postrain
    filegroup['negtrain'] = negtrain

    encoding_types = ['One_hot.csv', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                      'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                      'Micheletti_potentials', 'AESNN3', 'ANN4D']
    method_residual = []
    for encoding_type in encoding_types:
        for i in range(arg.begin, min_len + 1, arg.step):
            forward_methodname = "forward_" + str(i) + "_" + encoding_type
            backward_methodname = "backward_" + str(i) + "_" + encoding_type
            method_residual.append(forward_methodname)
            method_residual.append(backward_methodname)

    method_peptide = [
        "kmer", "DR.csv",  # 1
        "-PSSM-DT.csv", "-CC-PSSM.csv", "-AC-PSSM.csv", "-DT1.csv", "-Top-n-gram.csv", "-PDT-Profile.csv", "DP.csv",
        "ACC-PSSM.csv",  # 4
        "ACC.csv", "feature-AC.csv", "PDT.csv", "feature-CC.csv",  # 2
        "PC-PseAAC.csv", "SC-PseAAC.csv", "PC-PseAAC-General.csv", "SC-PseAAC-General.csv"  # 3
    ]
    if arg.feature_level == 'both':
        method = method_peptide + method_residual
    elif arg.feature_level == 'peptide':
        method = method_peptide
    elif arg.feature_level == 'residual':
        method = method_residual
    else:
        raise ValueError('No method type!')

    datadics = generate_data(filegroup, method)

    return datadics, data_len, seed


def cross_val(arg):
    cvdatadics = {}
    datadics, data_len, seed = get_feature(arg)
    setup_seed(seed)
    KF = KFold(n_splits=5, shuffle=True, random_state=seed)
    k = 1
    all_evaluation = []

    for train_index, test_index in KF.split(range(data_len)):
        method_names = []
        for methodname in datadics.keys():
            method_names.append(methodname)
            data = datadics[methodname]
            X_train, X_test = np.array(data[0])[train_index], np.array(data[0])[test_index]
            Y_train, Y_test = np.array(data[1], dtype=np.double)[train_index], np.array(data[1], dtype=np.double)[test_index]
            cvdatadics[methodname] = [X_train, Y_train, X_test, Y_test]

        print(f"the {k} th cross validation:")
        k = k + 1
        probability = train_ML_model(cvdatadics)

        train_dataset = ACPDataset(probability, train=True)
        test_dataset = ACPDataset(probability, train=False)

        max_epoch, best_evaluation = train(device, train_dataset, test_dataset, arg.batch_size, arg.epochs, lr=arg.lr)
        all_evaluation.append(best_evaluation)

    all_evaluation = pd.DataFrame(all_evaluation)

    return all_evaluation.mean(axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_text', default='ACP740/ACP740')
    parser.add_argument('--feature_level', default='both')
    parser.add_argument('--data_name', default='ACP240')
    parser.add_argument('--begin', default=1, type=int)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--end', default=11, type=int)
    parser.add_argument('--times', default=1, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    arg = parser.parse_args()

    if os.path.exists(f'{arg.data_name}_output') is False:
        os.mkdir(f'{arg.data_name}_output')

    arg.save_text = f'{arg.data_name}_output/{arg.data_name}_{arg.feature_level}_{arg.begin}_{arg.step}_{arg.end}'
    result = cross_val(arg=arg)

