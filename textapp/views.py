from django.shortcuts import render
from django.conf import settings

# Create your views here.
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression

# ============= TO DO!!! ==============
# - collect common functions into a machine_learning file and set of standard functions
# for Word Vectoristaion, Classifier initialisation etc

import pickle

senti_vect = pickle.load(open(settings.BASE_DIR + '/textapp/data/ngrams_sentiment_vect_20k_features.p', 'rb'))
senti_clf = pickle.load(open(settings.BASE_DIR + '/textapp/data/ngrams_sentiment_model_20k_features.p', 'rb'))

spami_vect = pickle.load(open(settings.BASE_DIR + '/textapp/data/spam_detector_vect_BoW_mindf5_ngramchars25.p', 'rb'))
spami_clf = pickle.load(open(settings.BASE_DIR + '/textapp/data/spam_detector_clfLR_BoW_mindf5_ngramchars25.p', 'rb'))


def index(request):
    template = 'textapp/index.html'

    return render(request, template)


def analyse_sentiment(request):
    # get text from the form submission
    X_test = request.POST['textToAnalyse']

    # transform into doc-term matrix (nparray)
    X_test_transformed = sent_vect.transform([X_test])
    # print(X_test_transformed)

    #
    prediction = senti_clf.predict(X_test_transformed)
    if prediction[0] == 1:
        prediction = 'Positive'
    else:
        prediction = 'Negative'

    print('Text = "' + X_test + '"\n' + 'Sentiment Prediction = ' + prediction)

    context = {
        'prediction': prediction,
    }

    template = 'textapp/index.html'

    return render(request, template, context)


def detect_spam(request):
    # get text from the form submission
    X_test = request.POST['spamToDetect']

    # transform into doc-term matrix (nparray)
    X_test_transformed = spami_vect.transform([X_test])
    # print(X_test_transformed)

    #
    prediction = spami_clf.predict(X_test_transformed)
    if prediction[0] == 1:
        prediction = 'Spam'
    else:
        prediction = 'Not Spam'

    print('Text = "' + X_test + '"\n' + 'Spam Prediction = ' + prediction)

    context = {
        'prediction': prediction,
    }

    template = 'textapp/spam.html'

    return render(request, template, context)


from .document_similarity import document_path_similarity


def calc_similarity(request):
    context = {}
    if request.POST:
        # get text from the form submission
        doc1 = request.POST['doc1']
        doc2 = request.POST['doc2']

        # calculate similarity
        similarity_score = document_path_similarity(doc1, doc2)

        print('Document similarity = ' + str(similarity_score))

        context = {
            'similarity_score': '{:.2f}'.format(similarity_score),
        }

    template = 'textapp/similarity.html'

    return render(request, template, context)
