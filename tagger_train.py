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


def prepare_sequence(seq, to_ix):  
    idxs = [to_ix[w] for w in seq]
    return idxs


def prepare_char_sequence(seq, to_ix):
  res = []
  for word in seq:
    res.append([to_ix[char] for char in word])


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim, conv_out_dim, hidden_dim, word_vocab_size, char_vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)

        kernel_size = 4
        self.char_conv = nn.Conv1d(char_embedding_dim, conv_out_dim, kernel_size)

        self.lstm = nn.LSTM(word_embedding_dim + char_embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def apply_conv(self, char_embeds):
        max_pooling_size = 3

        batch_s, sent_s, word_s, channel_s = char_embeds.shape
        char_embeds = char_embeds.view(batch_s, channel_s, sent_s, word_s)

        char_repr = self.char_conv(char_embeds)
        char_repr, _ = torch.max(char_repr, dim = max_pooling_size)

        batch_s, channel_s, word_s = char_embeds.shape
        char_embeds = char_embeds.view(batch_s, word_s, channel_s)  

        return char_embeds,batch_s,word_s

    def forward(self, words, chars):

        word_embeds = self.word_embeddings(words)
        char_embeds = self.char_embeddings(chars)

        char_embeds, batch_s, word_s = self.apply_conv(char_embeds)

        embeds = torch.cat((word_embeds, char_embeds), dim=2)

        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores.view(batch_s * word_s, -1)


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

    pad_word = len(word_to_ix)
    pad_char = len(char_to_ix)
    pad_tag = len(tag_to_ix)

    sentences.sort(key=lambda s: len(s[0]))

    WORD_EMBEDDING_DIM = 6
    CHAR_EMBEDDING_DIM = 6
    CONV_OUT_DIM = 3
    HIDDEN_DIM = 6

    model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, CONV_OUT_DIM, HIDDEN_DIM, 
                   len(word_to_ix) + 1, len(char_to_ix) + 1, len(tag_to_ix) + 1)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)



    print('Finished...')   


def extend_by_pads(chars_batch, pad):
    max_len = 0

    for sent in chars_batch:
      for word in sent:
        if len(word) > max_len:
          max_len = len(word)

    for sen in chars_batch:
      for word in sen:
        if len(word) < max_len:
          word.extend([pad] * (max_len - len(word)))   


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
