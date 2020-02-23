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

    return res

class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim, conv_out_dim, hidden_dim, word_vocab_size, char_vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)

        kernel_size = 4
        self.char_conv = nn.Conv1d(char_embedding_dim, conv_out_dim, kernel_size=(1, 4))

        # self.lstm = nn.LSTM(word_embedding_dim + char_embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(word_embedding_dim + conv_out_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def apply_conv(self, char_embeds):
        max_pooling_size = 3

        batch_s, sent_s, word_s, channel_s = char_embeds.shape
        char_embeds = char_embeds.view(batch_s, channel_s, sent_s, word_s)

        char_embeds = self.char_conv(char_embeds)
        char_embeds, _ = torch.max(char_embeds, dim=max_pooling_size)

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

def adjust_sym_batch(chars_batch, pad_char):
    max_len = 0

    for sen in chars_batch:
        # lens = np.array([len(w) for w in sen])
        # max_len = max(np.max(lens),max_len)
        # print(max_len)
        for word in sen:
            if len(word) > max_len:
                max_len = len(word)

    for sen in chars_batch:
        for word in sen:
            if len(word) < max_len:
                word.extend([pad_char] * (max_len - len(word)))


def one_epoch():
    pass

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

    batch_size = 4

    for epoch in range(10):
        train_index = 0

        while train_index < len(sentences):
            sentence_batch = []
            symbols_batch = []
            target_batch = []

            # manually create batches
            for i in range(batch_size):
                if train_index + i == len(sentences):
                    break

                sentence_batch.append(prepare_sequence(sentences[train_index + i][0], word_to_ix))
                symbols_batch.append(prepare_char_sequence(sentences[train_index + i][0], char_to_ix))
                target_batch.append(prepare_sequence(sentences[train_index + i][1], tag_to_ix))

            train_index += batch_size

            # pad sentence and word batches
            curr_len = len(sentence_batch[-1])

            if not all(len(x) == len(sentence_batch[-1]) for x in sentence_batch):
                for word_seq, char_seq, tag_seq in zip(sentence_batch, symbols_batch, target_batch):
                    while len(word_seq) != curr_len:
                        word_seq.append(pad_word)
                        tag_seq.append(pad_tag)
                        char_seq.append([pad_char])

            adjust_sym_batch(symbols_batch, pad_char)

            # print(sentence_batch)
            # print(symbols_batch)
            # print(target_batch)

            sen_in = torch.tensor(sentence_batch, dtype=torch.long)
            sym_in = torch.tensor(symbols_batch, dtype=torch.long)
            targets = torch.tensor(target_batch, dtype=torch.long)

            model.zero_grad()

            tag_scores = model(sen_in, sym_in)

            loss = loss_function(tag_scores.squeeze(1), targets.view(len(sentence_batch) * curr_len, ))
            loss.backward()
            optimizer.step()
        print("Epoch " + str(epoch))


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
