# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_model(train_file, model_file):
    words, tags = words_tags_split(train_file)
    print('Finished...')


def words_tags_split(train_file):
    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    words = []
    tags = []

    for train_line in train_lines:
        cur_train_pairs = train_line.strip().split(' ')
        for pair in cur_train_pairs:
            splitted = pair.split('/')
            words.append(splitted[0])
            tags.append(splitted[1])    

    return words, tags        


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
