#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Bio import SeqIO
from Bio.Seq import Seq
import json
import argparse
import numpy as np
import pandas as pd
import os 
import glob
def main():    
    data_path = "E:/AcrData/NAR_PaCRISPR_Datasets/Pa*/*.fasta"
    data_path = "biodata/dataset_acp240/*.fasta"
    # data_path = "D:/EnACP-GA-py3.6/ACP/Anti-cancer-data/iacp/*.fasta"
    data_path = "D:/EnACP-GA-py3.6/ACP/Anti-cancer-data/dataset_lee/*.fasta"
    # data_path = "biodata/dataset_acp740/*.fasta"
    filegroup = glob.glob(data_path)
    min_len = cal_Min_len_filegroup(filegroup)

    encoding_types = ['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix',
                      'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies',
                      'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec']
    # encoding_types = ['One_hot', 'One_hot_6_bit']
    begin = int(min_len/2)
    begin = int(min_len/2)
    begin = 1
    for i in range(begin,min_len+1,1):
        extract_features(filegroup,i,encoding_types)
        print(i)

    # print(os.getcwd())
    # filegroup = ["PaCRISPR_Training_Dataset/PaCRISPR_Training_Negative_902.fasta","PaCRISPR_Training_Dataset/PaCRISPR_Training_Positive_Original_488.fasta","PaCRISPR_Independent_Dataset/",""]
    # encoding_type = "ProtVec"
    # tmpfile = "valid_nonAVP45.fasta"

    #feature_extraction(tmpfile,encoding_type,True)
def mkd(x):
    if(os.path.exists(x)):
        pass
    else:
        os.makedirs(x)
def extract_features(filegroup,min_len,encoding_types):
    # a = path+"/method-feature/"
    for encoding in encoding_types:
        for filepath in filegroup:
            feature_extraction(filepath,min_len,encoding)


def feature_extraction(tmpfile,min_len,encoding_type="One_hot"):
    overlap = True
    fasta_file = open(tmpfile, "r")
    path,filename = os.path.split(tmpfile)
    method_path = path+"/new-method-feature/"
    mkd(method_path)
    if (os.path.exists(method_path+filename+"forward_"+str(min_len)+"_"+encoding_type+".csv") and os.path.exists(method_path+filename+"backward_"+str(min_len)+"_"+encoding_type+".csv")):
        print(encoding_type+"has existed")
        return 0
    forward_feature = []
    backward_feature = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = seq_record.seq
        # print(len(seq))
        # print(seq)
        forward_str = seq[0:min_len]
        # print(forward_str)
        backward_str = seq[len(seq)-min_len:len(seq)+1]
        # 序列翻转
        backward_str = backward_str[::-1]
        # print(backward_str)
        seqEncoding = SequenceEncoding(encoding_type)
        forward_encodings = seqEncoding.get_encoding(forward_str,overlap)
        backward_encodings = seqEncoding.get_encoding(backward_str,overlap)
    # print (encodings)
    # print(type(encodings))
        forward_feature_vector = []
        backward_feature_vector = []

        for i in forward_encodings:
            for j in i:
                forward_feature_vector+=i[j]
        for i in forward_encodings:
            for j in i:
                backward_feature_vector+=i[j]
        #feature_vector = np.array(feature_vector)
        forward_feature.append(forward_feature_vector)
        backward_feature.append(backward_feature_vector)
    #print (feature)
    forward_feature = pd.DataFrame(forward_feature)
    backward_feature = pd.DataFrame(backward_feature)
    forward_feature.to_csv(method_path+filename+"forward_"+str(min_len)+"_"+encoding_type+".csv",index=False,header=None)

    backward_feature.to_csv(method_path+filename+"backward_"+str(min_len)+"_"+encoding_type+".csv",index=False,header=None)

def cal_Min_len_filegroup(filegroup):
    Min_len_list = []
    for file in filegroup:
        len_ = cal_Min_len(file)
        print (len_)
        Min_len_list.append(len_)
    return min(Min_len_list)
def cal_Min_len(filepath):
    fasta_file = open(filepath, "r")
    len_list = []
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seq = seq_record.seq
        len_list.append(len(seq))
    return min(len_list)

class SequenceEncoding:
    encoding_types = ['One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix', 
                      'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies', 
                      'Micheletti_potentials', 'AESNN3', 'ANN4D', 'ProtVec']
    residue_types = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
    def __init__(self, encoding_type="One_hot"):
        if encoding_type not in SequenceEncoding.encoding_types:
            raise Exception("Encoding type \'%s\' not found" % encoding_type)
        self.encoding_type = encoding_type

    def get_ProtVec_encoding(self, ProtVec, seq, overlap=True):
        if overlap:
            encodings = []
            for i in range(len(seq)-2):
                encodings.append({seq[i:i+3]: ProtVec[seq[i:i+3]]}) if ProtVec.__contains__(seq[i:i+3]) else encodings.append({seq[i:i+3]: ProtVec["<unk>"]})
        else:
            encodings_1, encodings_2, encodings_3 = [], [], []
            for i in range(0, len(seq), 3):
                if i+3 <= len(seq):
                    encodings_1.append({seq[i:i+3]: ProtVec[seq[i:i+3]]}) if ProtVec.__contains__(seq[i:i+3]) else encodings_1.append({seq[i:i+3]: ProtVec["<unk>"]})
                if i+4 <= len(seq):
                    encodings_2.append({seq[i+1:i+4]: ProtVec[seq[i+1:i+4]]}) if ProtVec.__contains__(seq[i+1:i+4]) else encodings_2.append({seq[i+1:i+4]: ProtVec["<unk>"]})
                if i+5 <= len(seq):
                    encodings_3.append({seq[i+2:i+5]: ProtVec[seq[i+2:i+5]]}) if ProtVec.__contains__(seq[i+2:i+5]) else encodings_3.append({seq[i+2:i+5]: ProtVec["<unk>"]})
                
            encodings = [encodings_1, encodings_2, encodings_3]
        return encodings
        
    def get_encoding(self, seq, overlap=True):
        seq = seq.upper()
        with open("data/%s.json" % self.encoding_type, 'r') as load_f:
            encoding = json.load(load_f)
        encoding_data = []
        if self.encoding_type == "ProtVec":            
            encoding_data = self.get_ProtVec_encoding(encoding, seq, overlap)
        else:
            for res in seq:
                if res not in SequenceEncoding.residue_types:
                    res = "X"
                encoding_data.append({res: encoding[res]})
                    
        return encoding_data

        
if __name__ == '__main__':
    main()