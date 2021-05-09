import numpy as np
import json
import normalization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = open('ziniu_baze.json', encoding="utf8").read()


def plot_embedding(X, title=None):
    colors = ['red','green','blue','purple', 'dimgray','mediumpurple','orange','red','brown','gray', 'pink','limegreen','royalblue','tan','navy','yellowgreen','gold','teal','coral','maroon','crimson','plum','rosybrown','olivedrab','khaki','silver','indigo','green','green','green','green','green','green','green','green']
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=colors[Y[i]],
                 fontdict={'weight': 'bold', 'size': 11})
    plt.title(title)


klase = []
uzklausa = []
decoder = []
new_pattern = ""
i=1

intents = json.loads(data)
for intent in intents['intents']:
    for pattern in intent['uzklausos']:
        klase.append(intent['tag'])
        for word in pattern.split():
            normalized_word = normalization.lem(word.lower())
            if normalized_word not in normalization.ignore_words:
                new_pattern = new_pattern + ' ' + normalized_word
        uzklausa.append(new_pattern)
        new_pattern = ""
    decoder.append((intent['tag'], i))
    i += 1

skaicius_klase = []

for tg, sk in decoder:
    for kl in klase:
        if(kl == tg):
            skaicius_klase.append(sk)

X = uzklausa
Y = skaicius_klase
vect = TfidfVectorizer()
# vect = CountVectorizer()
X = vect.fit_transform(X)

size_train = X.shape[0]
X_tsne = TSNE(n_components=2, random_state=0).fit_transform(X) 
plot_embedding(X_tsne, "Vektorizuotos užklausos")
plt.show()
