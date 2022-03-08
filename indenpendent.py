# -*- coding: UTF-8 -*-
import glob
import numpy as np
import pandas as pd
import torch
from dataset import ACPDataset
from train import train
import argparse
import os
from lightgbm.sklearn import LGBMClassifier

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_feature(arg):
    if arg.data_name == 'Main':
        tmpdir = "Anti-cancer-data/Main"
        postrain = glob.glob(tmpdir + '/ACP20mainTrain-Pos.fasta*')
        negtrain = glob.glob(tmpdir + '/ACP20mainTrain-Neg.fasta*')
        postest = glob.glob(tmpdir + '/ACP20mainTest-Pos.fasta*')
        negtest = glob.glob(tmpdir + '/ACP20mainTest-Neg.fasta*')
        min_len = 3
        seed = 9
    elif arg.data_name == 'Alternate':
        tmpdir = "Anti-cancer-data/Alt"
        postrain = glob.glob(tmpdir + '/ACP20AltTrain-Pos.fasta*')
        negtrain = glob.glob(tmpdir + '/ACP20AltTrain-Neg.fasta*')
        postest = glob.glob(tmpdir + '/ACP20AltTest-Pos.fasta*')
        negtest = glob.glob(tmpdir + '/ACP20AltTest-Neg.fasta*')
        min_len = 3
        seed = 5
    else:
        raise ValueError('No dataset name!')
    filegroup = {}
    filegroup['postrain'] = postrain
    filegroup['negtrain'] = negtrain
    filegroup['postest'] = postest
    filegroup['negtest'] = negtest

    encoding_types = ['One_hot.csv', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                      'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                      'Micheletti_potentials', 'AESNN3', 'ANN4D']
    method_peptide = ["-DT1.csv", "-PDT-Profile.csv", "-Top-n-gram.csv", "-CC-PSSM.csv", "-AC-PSSM.csv", '-AC.csv',
                      "ACC-PSSM.csv", "kmer", "feature-AC.csv", "ACC.csv", "feature-CC.csv", "DP.csv", "DR.csv",
                      "PC-PseAAC.csv", "PC-PseAAC-General.csv", "PDT.csv", "-PSSM-DT.csv", "SC-PseAAC-General.csv",
                      "SC-PseAAC.csv"]
    method_residual = []
    for encoding_type in encoding_types:
        for i in range(arg.begin, min_len + 1, arg.step):
            forward_methodname = "forward_" + str(i) + "_" + encoding_type
            backward_methodname = "backward_" + str(i) + "_" + encoding_type
            method_residual.append(forward_methodname)
            method_residual.append(backward_methodname)

    if arg.feature_level == 'both':
        method = method_peptide + method_residual
    elif arg.feature_level == 'peptide':
        method = method_peptide
    elif arg.feature_level == 'residual':
        method = method_residual
    else:
        raise ValueError('No method type!')

    datadics = generate_data(filegroup, method)
    setup_seed(seed)
    data = train_ML_model(datadics)

    return data, seed


def generate_data(filegroup, method):
    postrain = filegroup["postrain"]
    negtrain = filegroup["negtrain"]
    postest = filegroup["postest"]
    negtest = filegroup["negtest"]
    method_data = {}
    for method_name in method:
        for i in postrain:
            if method_name in i:
                postrain_method = i
                break
        for j in negtrain:
            if method_name in j:
                negtrain_method = j
                break
        for k in postest:
            if method_name in k:
                postest_method = k
                break
        for l in negtest:
            if method_name in l:
                negtest_method = l
                break

        filepath = [negtest_method, negtrain_method, postest_method, postrain_method]

        method_data[method_name] = file_reading(filepath)
    return method_data


def file_reading(filepath):
    dataset1 = pd.read_csv(filepath[0], header=None, low_memory=False, dtype=np.float32)
    dataset2 = pd.read_csv(filepath[1], header=None, low_memory=False, dtype=np.float32)
    dataset3 = pd.read_csv(filepath[2], header=None, low_memory=False, dtype=np.float32)
    dataset4 = pd.read_csv(filepath[3], header=None, low_memory=False, dtype=np.float32)

    # only be used except for lightGBM
    # dataset1 = dataset1.fillna(0)
    # dataset2 = dataset2.fillna(0)
    # dataset3 = dataset3.fillna(0)
    # dataset4 = dataset4.fillna(0)

    train_data = pd.concat([dataset2, dataset4], axis=0)
    test_data = pd.concat([dataset1, dataset3], axis=0)

    neg_train_tags = [0.0] * dataset2.shape[0]
    pos_train_tags = [1.0] * dataset4.shape[0]
    train_tags = neg_train_tags + pos_train_tags

    neg_test_tags = [0.0] * dataset1.shape[0]
    pos_test_tags = [1.0] * dataset3.shape[0]
    test_tags = neg_test_tags + pos_test_tags

    data = [train_data, train_tags, test_data, test_tags]
    return data


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


def independent(arg):
    all_evaluation = []
    probability, seed = generate_feature(arg)
    setup_seed(seed)

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
    result = independent(arg=arg)
