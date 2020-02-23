# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
import numpy as np
import string
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def words_tags_split(train_file):
    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    sentences = []

    for train_line in train_lines:

        words = []
        tags = []

        cur_train_pairs = train_line.strip().split(' ')
        for pair in cur_train_pairs:
            splitted = pair.rsplit('/', 1)
            words.append(splitted[0])
            tags.append(splitted[1])   

        sentences.append((words,tags))     

    return sentences


def data_idx(sentences):
    word_to_ix = {}
    char_to_ix = {}
    for words,tags in sentences:
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)

    return word_to_ix, char_to_ix

def train_model(train_file, model_file):
    # parse data
    sentences = words_tags_split(train_file)

    # generate tags set
    tags_set = set()
    for words,tags in sentences:
        for tag in tags:
            tags_set.add(tag)

    # create indexes for words and chars
    word_to_ix, char_to_ix = data_idx(sentences)

    # create indexes for tags
    tag_to_ix = {w: i for i, w in enumerate(list(tags_set))}
    ix_to_tag = {i: w for i, w in enumerate(list(tags_set))}

    sentences.sort(key=lambda s: len(s[0]))

    # generate char embeddings
    CHAR_EMBEDDING_DIM = 6
    CHAR_RADIUS = 3

    # sos is a start of sequence, eos is an end of sequence
    vocab = list(string.printable)
    char_embed = nn.Embedding(len(vocab), CHAR_EMBEDDING_DIM)

    word_embeddings = nn.Embedding()

    print('Finished...')      


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
