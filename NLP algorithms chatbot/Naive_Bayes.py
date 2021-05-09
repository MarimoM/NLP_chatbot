import nltk
import json
import datetime
import numpy as np
import normalization
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

data = open('train_1.json', encoding="utf8").read()
test_data = open('test_1.json', encoding="utf8").read()
ERROR_THRESHOLD = 0.01

def best_parameter():
    pipeline = Pipeline([
        # ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', OneVsRestClassifier(MultinomialNB())),
    ])
    parameters = {
        # 'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__alpha': (1, 5, 1e-1, 1e-2, 1e-3)
    }

    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
    grid_search_tune.fit(X, Y)

    print("Geriausi parametrai:")
    print (grid_search_tune.best_estimator_.steps)

def normalize(data):
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

    return uzklausa, skaicius_klase, decoder

X, Y, decoder = normalize(data)
vect = TfidfVectorizer(ngram_range=(1, 2))
X = vect.fit_transform(X)

startTime = datetime.datetime.now()
clf = MultinomialNB(alpha=5, fit_prior=True).fit(X, Y)
print("Apmokymo laikas: ", (datetime.datetime.now() - startTime).microseconds)

# best_parameter()

testavimo_laikas = []
correct = 0
count = 0
test_intents = json.loads(test_data)

for intent in test_intents['intents']:
    for pattern in intent['uzklausos']:
        sentence_words = nltk.word_tokenize(pattern)
        sentence_words = [normalization.lem(w.lower()) for w in sentence_words]
        sentence_words = [w.lower() for w in sentence_words if w not in normalization.ignore_words]
        new_pattern = ""

        for word in sentence_words:
            new_pattern = new_pattern + ' ' + word

        startTime = datetime.datetime.now()
        result = clf.predict(vect.transform([new_pattern]).toarray())
        testavimo_laikas.append(datetime.datetime.now() - startTime)

        probabilities = clf.predict_proba(vect.transform([new_pattern]).toarray())

        for prob in probabilities:
            max_value = max(prob)
            max_index = np.where(prob == max_value)

        print(max_value)

        expected = intent['tag']
        actual = ""
        if (max_value <= ERROR_THRESHOLD):
            actual = "kita"

        if not actual:
            for tg, sk in decoder:
                if(sk == result):
                    actual = tg

        if(actual == expected):
            correct += 1
        print(actual, expected)
        count += 1

print(100 * correct / count, "% correct answers")

testavimoLaikoSuma = 0.0
for e in testavimo_laikas:
    testavimoLaikoSuma += e.microseconds

print("Testavimo laikas: ", testavimoLaikoSuma)