from django.shortcuts import render
from django.conf import settings

# Create your views here.
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
import pickle

vect = pickle.load(open(settings.BASE_DIR + '/textapp/data/ngrams_sentiment_vect_20k_fetures.p', 'rb'))
clf = pickle.load(open(settings.BASE_DIR + '/textapp/data/ngrams_sentiment_model_20k_fetures.p', 'rb'))


def index(request):
    template = 'textapp/index.html'

    return render(request, template)


def analyse_sentiment(request):
    # get text from the form submission
    X_test = request.POST['textToAnalyse']

    # transform into doc-term matrix (nparray)
    X_test_transformed = vect.transform([X_test])
    # print(X_test_transformed)

    #
    prediction = clf.predict(X_test_transformed)
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
