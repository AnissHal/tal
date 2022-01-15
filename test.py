import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import string
import re
import sys

# Chargement du vocabulaire utiliser par le model
def read_vocab(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


# Si un mot est non existant, une erreur sera retournée
word2id = read_vocab('vocab')

id2word = {id: token for id, token in word2id.items()}
label_vocab = ['negative', 'positive']

# Nettoie le texte et vectorise
def vectorize(text):
    text = text.lower()
    text = re.sub(r'[,.!;()]', r' ', text)
    one_hot = torch.zeros(len(word2id))
    for token in text.split(' '):
        if token not in string.punctuation:
            index = id2word[token]
            one_hot[index] = 1

    return one_hot


class ReviewClassifier(nn.Module):
    def __init__(self, n_feature):
        super(ReviewClassifier, self).__init__()

        self.lf = nn.Linear(n_feature, 1, dtype=torch.float32)

    def forward(self, x):
        out = self.lf(x)
        out = F.sigmoid(out)
        return out


model = ReviewClassifier(len(word2id))
model.load_state_dict(torch.load('final_model'))

model.eval()
print(label_vocab[round(model(vectorize(sys.argv[1])).item())])
