# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

def words_split(train_file):
    reader = open(train_file)
    train_lines = reader.readlines()
    reader.close()

    sentences = []

    for train_line in train_lines:

        words = train_line.strip().split(' ')

        sentences.append(words)     

    return sentences


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim, conv_out_dim, hidden_dim, word_vocab_size, char_vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)

        kernel_size = 4
        self.char_conv = nn.Conv1d(char_embedding_dim, conv_out_dim, kernel_size=(1, 4))

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


def prepare_sequence(seq, to_ix):  
    idxs = [to_ix[w] for w in seq]
    return idxs


def prepare_char_sequence(seq, to_ix):
    res = []
    for word in seq:
        res.append([to_ix[char] for char in word])

    return res

def extend_by_pads(chars_batch, pad):
    max_len = 0
    for sent in chars_batch:
        lens = np.array([len(word) for word in sent])
        max_len = max(np.max(lens),max_len)


    for sen in chars_batch:
        for word in sen:
            if len(word) < max_len:
                word.extend([pad] * (max_len - len(word)))

def tag_sentence(test_file, model_file, out_file):
    
    sentences = words_split(test_file)
    
    word_to_ix, char_to_ix, ix_to_tag, model_state_dict = torch.load(model_file)

    pad_char = len(char_to_ix)
    
    WORD_EMBEDDING_DIM = 6
    CHAR_EMBEDDING_DIM = 6
    CONV_OUT_DIM = 3
    HIDDEN_DIM = 6

    model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, CONV_OUT_DIM, HIDDEN_DIM, 
                len(word_to_ix) + 1, len(char_to_ix) + 1, len(ix_to_tag) + 1)

    model.load_state_dict(model_state_dict)       

    for sent in sentences:
        sent_prepared = [prepare_sequence(sent, word_to_ix)]
        chars_prepared = [prepare_char_sequence(sent, char_to_ix)]
    
        extend_by_pads(chars_prepared, pad_char)

        sen_in = torch.tensor(sent_prepared, dtype=torch.long)
        sym_in = torch.tensor(chars_prepared, dtype=torch.long)
        tag_scores = model(sen_in, sym_in)
        print(tag_scores.squeeze(1))
        exit(0)
    # tag_scores = model(sen_in, sym_in)

    print('Finished...')




if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
