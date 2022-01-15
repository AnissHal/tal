from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import string
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import re
import pandas as pd
import sys


def read_data(ntrain=None, ntest=None):
    train_filename = sys.argv[2] if len(sys.argv) > 1 else 'mv_train.csv'
    test_filename = sys.argv[3] if len(sys.argv) > 1 else 'mv_test.csv'
    return pd.read_csv(train_filename, nrows=ntrain), pd.read_csv(test_filename, nrows=ntest)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'([.,?!])', r" \1", text)
    text = re.sub(r'\\', r' ', text)
    text = re.sub(r'<br />', r' ', text)
    text = re.sub(r'^[a-zA-Z.,?!]', r' ', text)
    return text

class ReviewDataset(Dataset):
    def __init__(self, df, vocab, label_vocab, vectorize):
        self.df = df
        self.vocab = vocab
        self.label_vocab = label_vocab
        self._vectorize = vectorize

    def __len__(self):
        return len(self.df['text'])

    def __getitem__(self, index):
        row = self.df.iloc[index]
        one_hot = self._vectorize(row['text'])

        return {'x': one_hot,
                'y': torch.FloatTensor([row['label']])}

# Lookup table pour vectoriser
class Dictionnaire(object):
    def __init__(self, word2id=None):
        self.word2id = word2id
        if self.word2id is None:
            self.word2id = {}

        self.id2word = {id: token for id, token in self.word2id.items()}

        self.unk_token = self.add_token('<UNK>')

        self.unk_index = self.id_by_token('<UNK>')

    def add_token(self, token):
        if token not in self.word2id:
            index = len(self.word2id)
            self.word2id[token] = index
            self.id2word[index] = token
        return self

    def id_by_token(self, token):
        if token not in self.word2id:
            return self.unk_index
        else:
            return self.word2id[token]

    def token_by_id(self, id):
        return self.id2word[id]

    def __len__(self):
        return len(self.word2id)

    @classmethod
    def from_df(cls, df):
        vocab = Dictionnaire()
        for doc in df['text'].values:
            for token in doc.split(" "):
                if token not in string.punctuation:
                    vocab.add_token(token)
        return vocab

    def save_vocab(self, path):
        file = open('vocab', 'wb')
        pickle.dump(self.word2id, file)
        file.close()


class Vectorizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def vectorize(self, row):
        one_hot = torch.zeros(len(vocab), dtype=torch.float32)

        # Vectorisation en collapsed vector (Par phrase)
        for word in row.split(" "):
            if word not in string.punctuation:
                one_hot[self.vocab.id_by_token(word)] = 1

        return one_hot

# Perceptron de classification binaire
class ReviewClassifier(nn.Module):
    def __init__(self, n_feature):
        super(ReviewClassifier, self).__init__()

        self.lf = nn.Linear(n_feature, 1, dtype=torch.float32)

    # Utiliser une fonction d'activation sigmoid ou softmax
    def forward(self, x):
        out = self.lf(x)
        out = F.sigmoid(out)
        return out


total_test = 3000

data, test = read_data(40000, total_test)

data['text'] = data['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Creation de vocabulaire
vocab = Dictionnaire.from_df(pd.concat([data, test]))
vectorizer = Vectorizer(vocab)

label_vocab = Dictionnaire()
label_vocab.add_token('negative').add_token('positive')

vocab.save_vocab('vocab')

dataset = ReviewDataset(data, vocab, label_vocab, vectorizer.vectorize)

dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

# Hyperparamètres
l_rate = 0.010

epoch = 15

classifier = ReviewClassifier(len(vocab))

bce_loss = nn.BCEWithLogitsLoss()

optimizer = Adam(classifier.parameters(), lr=l_rate)


# Apprentissage
for n_epoch in range(epoch):
    for i, batch in enumerate(dataloader):
        pred = classifier(batch['x']).float()

        optimizer.zero_grad()

        loss = bce_loss(pred, batch['y'])

        loss.backward()

        optimizer.step()

        if i % 1000 == 0:
            print("epoch {}, loss: {}".format(n_epoch, loss.item()))

try:
    torch.save(classifier.state_dict(), sys.argv[1])
except:
    raise Exception("Impossible de sauvegarder le model, vérifier si vous avez passez l'argument")

classifier.eval()

# Test
dataset = ReviewDataset(test, vocab, label_vocab, vectorizer.vectorize)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

acc = 0

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        pred = classifier(batch['x']).float()

        rounded = torch.round(pred)

        acc += torch.sum(rounded == batch['y'])

print("test accuracy: {}".format(acc / total_test))
