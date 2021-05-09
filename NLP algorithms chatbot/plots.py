import nltk
import json
import numpy as np
import normalization
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm
from itertools import cycle
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import learning_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold

data = open('ziniu_baze.json', encoding="utf8").read()
test_data = open('ziniu_baze.json', encoding="utf8").read()

def get_roc_curve(train_data_set, test_data_set, train_answers, test_answers, clf):
    clf.fit(train_data_set, train_answers)
    score = clf.predict_proba(test_data_set)
    sc = clf.predict(test_data_set)
    acc = accuracy_score(test_answers, sc)
    print(acc)

    ns_probs = [0 for _ in range(len(test_answers))]

    lb = LabelBinarizer()
    lb.fit(test_answers)
    y_test = lb.transform(test_answers)
    ns_probs = lb.transform(ns_probs)

    n_classes = score.shape[1]

    nc = roc_auc_score(y_test, ns_probs)
    roc_auc = roc_auc_score(y_test, score,  multi_class='ovr')

    # print('Logistic: ROC AUC=%.3f' % (roc_auc))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr["micro"], tpr["micro"], tr = roc_curve(y_test.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr['micro'], tpr['micro'], roc_auc['micro']

def plot_learning_curve(estimator, title, X, y, cv=None):
    plt.figure()
    plt.title(title)
    plt.xlabel("Apmokymo rinkinio dydis")
    plt.ylabel("Tikslumas")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, train_sizes = [26, 52, 78, 104, 130, 156, 182, 208, 234, 260, 286, 312], shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Apmokymo rezultatas")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Kryžminės patikros rezultatas")

    plt.ylim([0.0, 1.1])

    plt.legend(loc="best")
    return plt

def normalize(data):
    klase = []
    uzklausa = []
    decoder = []
    new_pattern = ""
    i=0
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

def box_plots():
    dataANN = [0.90, 0.91, 0.95, 0.96, 0.96]
    dataKNN = [90, 91, 95, 89, 87]
    dataSVM = [91, 97, 93, 98, 95]
    dataNB = [93, 95, 95, 93, 90]
    data = [dataANN, dataKNN, dataSVM, dataNB]
    fig = plt.figure(figsize =(10, 7))
    plt.title('Klasifikavimo algoritmų tikslumo palyginimas')
    plt.boxplot(data)
    plt.xticks([1, 2, 3, 4], ["DNN", "KNN", "SVM", "NB"])
    plt.show()

def roc(clf, title):

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    i = 0

    for train_index, test_index in kf.split(X, Y):
        X_train_set, X_test_set = X[train_index], X[test_index]
        y_train_answers, y_test_answers = np.array(Y)[train_index], np.array(Y)[test_index]
        fpr[i], tpr[i], roc_auc[i] = get_roc_curve(X_train_set, X_test_set, y_train_answers, y_test_answers, clf)
        i +=1

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 5

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    lw = 1

    plt.figure()
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='Pirmas rinkinys, AUC: %0.3f' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='aqua', lw=lw, label='Antras rinkinys, AUC: %0.3f' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], color='green', lw=lw, label='Trečias rinkinys, AUC: %0.3f' % roc_auc[2])
    plt.plot(fpr[3], tpr[3], color='red', lw=lw, label='Ketvirtas rinkinys, AUC: %0.3f' % roc_auc[3])
    plt.plot(fpr[4], tpr[4], color='cornflowerblue', lw=lw, label='Penktas rinkinys, AUC: %0.3f' % roc_auc[4])
    plt.plot(fpr["macro"] ,  tpr["macro"], 'k--', color='navy', lw=2, label='Vidurkis, AUC: %0.3f' % roc_auc["macro"])
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.00, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Specifiškumas')
    plt.ylabel('Jautrumas')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def learning_curves(clf, title):
    plot_learning_curve(clf, title, X, Y, cv=5)
    plt.show()

X, Y, decoder = normalize(data)
vect = TfidfVectorizer(ngram_range=(1, 2))
X = vect.fit_transform(X)

# KNN
knn_clf = KNeighborsClassifier(n_neighbors=5).fit(X, Y)
learning_curves(knn_clf, 'Mokymosi kreivė (K arčiausių kaimynų)')
roc(knn_clf, title="ROC kreivė (K arčiausių kaimynų)")

# SVM
svm_clf = svm.SVC(tol=1e-2, max_iter=100, kernel='linear').fit(X, Y)
learning_curves(svm_clf, 'Mokymosi kreivė (Atraminių vektorių klasifikatorius)')
roc(svm_clf, title="ROC kreivė (Atraminių vekrotių mašina)")

# MLP
mlp_clf = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', max_iter=50, alpha=0.01, tol=1e-9).fit(X, Y)
learning_curves(mlp_clf, 'Mokymosi kreivė (Daugiasluoksnis perceptronas)')
roc(mlp_clf, title="ROC kreivė (Daugiasluoksnis perceptronas)")

# NB
nb_clf = MultinomialNB(alpha=5, fit_prior=True).fit(X, Y)
learning_curves(nb_clf, 'Mokymosi kreivė (Naivusis Bajesas)')
roc(nb_clf, title="ROC kreivė (Naivusis Bajesas)")

# box_plots()