# coding: utf-8
'''
This is a python3 file
'''

from collections import OrderedDict

class Tokenizer(object):
    def __init__(self, vocab_file,stopwords=set('.。,，?？!！')):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        self.stopwords = stopwords
    def tokenize(self, sentence):
        tokens =  sentence.split(' ')
        res = []
        for token in tokens:
            for s in self.stopwords:
                token = token.replace(s, '')
            if token:
                res.append(token)
        return res
    def convert_tokens_to_ids(self, tokens, seq_length=None):
        res = [2]
        if seq_length:
            for token in tokens[:seq_length-1]:
                if not token:
                    continue
                if token in self.vocab:
                    res.append(self.vocab[token])
                else:
                    res.append(self.vocab['[UNK]'])
            if len(res)<seq_length:
                res.append(3)
            res += (seq_length - len(res)) * [self.vocab['[PAD]']]
        else:
            for token in tokens:
                if not token:
                    continue
                if token in self.vocab:
                    res.append(self.vocab[token])
                else:
                    res.append(self.vocab['[UNK]'])
        return res

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[temp] for temp in ids]

def load_vocab(vocab_file):
    vocab = OrderedDict()
    index = 0
    vocab['[PAD]'] = index
    index += 1
    vocab['[UNK]'] = index
    index += 1
    vocab['[GO]'] = index
    index += 1
    vocab['[EOS]'] = index
    index += 1
    for k in vocab_file:
        vocab[k] = index
        index += 1
    return vocab

if __name__ == '__main__':
    pass
