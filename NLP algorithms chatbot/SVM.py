import nltk
import json
import datetime
import numpy as np
import pandas as pd
import normalization
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def best_parameter():
    pipeline = Pipeline([
    # ('tfidf', TfidfVectorizer()),
    ('clf', svm.LinearSVC()),
    ])
    parameters = {
    # 'tfidf__ngram_range': [(1, 2)],
    'clf__C': [1, 10, 100, 1000, 0.1],
    'clf__tol': (1e-1, 1e-4),
    }

    grid_search_tune = GridSearchCV(pipeline, parameters)
    grid_search_tune.fit(X, Y)

    print("Geriausi parametrai:")
    print(grid_search_tune.best_estimator_)

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

data = open('train_4.json', encoding="utf8").read()
test_data = open('test_4.json', encoding="utf8").read()
ERROR_THRESHOLD = 0.8

X, Y, decoder = normalize(data)
vect = TfidfVectorizer(ngram_range=(1, 2))
X = vect.fit_transform(X)

startTime = datetime.datetime.now()
clf =  OneVsRestClassifier(svm.SVC(tol=1e-2, max_iter=50, kernel='linear', probability=True)).fit(X, Y)
# clf =  OneVsRestClassifier(svm.LinearSVC(tol=0.01, random_state=0)).fit(X, Y)
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
        r= clf.predict_proba(vect.transform([new_pattern]).toarray())
        testavimo_laikas.append(datetime.datetime.now() - startTime)

        # score = clf.decision_function(vect.transform([new_pattern]).toarray())

        for sc in r:
            max_value = max(sc)
            max_index = np.where(sc == max_value)

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

testavimoLaikasSume = 0.0
for t in testavimo_laikas:
    testavimoLaikasSume += t.microseconds

print("Testavimo laikas: ", testavimoLaikasSume)
