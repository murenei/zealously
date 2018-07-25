from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponseRedirect

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


def sentiment(request):
    template = 'textapp/sentiment.html'

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

    template = 'textapp/sentiment.html'

    return render(request, template, context)


def spam(request):
    template = 'textapp/spam.html'

    return render(request, template)


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
from .forms import SimilarityForm
from django.views import View


def similarity(request):
    template = 'textapp/similarity.html'

    return render(request, template)


class CalcSimilarity(View):
    form = SimilarityForm(initial={'doc1': 'Dump some text here!'})
    template = 'textapp/similarity.html'

    def get(self, request):
        return render(request, self.template, {'form': self.form})

    def post(self, request):
        # get text from the form submission
        doc1 = request.POST['doc1']
        doc2 = request.POST['doc2']

        print(doc1, doc2)

        # calculate similarity
        similarity_score = document_path_similarity(doc1, doc2)

        print('Document similarity = ' + str(similarity_score))

        # self.form.fields["doc1"].initial = doc1

        context = {
            'similarity_score': '{:.2f}'.format(similarity_score),
            'form': self.form,
        }
        return render(request, self.template, context)


# ========== DETECT FACES ==========


# Imaginary function to handle an uploaded file.
# from somewhere import handle_uploaded_file
from .forms import UploadFileForm
import os

from django.conf import settings
UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, 'uploads')

# IMAGE PROCESSING DEPENDENCIES
import cv2
import numpy as np
import base64


class DetectFaces(View):

    template = 'textapp/image.html'
    form = UploadFileForm()

    def get(self, request):

        return render(request, self.template, {'form': self.form})

    def post(self, request):
        print(request.POST)
        print(request.FILES)
        print(request.FILES['file'])

        self.form = UploadFileForm(request.POST, request.FILES)

        if self.form.is_valid():
            print('VALID!')
            # handle_uploaded_file(request.FILES['file'])
            faceDetected, num_faces, to_send, facefilename = upload_file(request.FILES['file'])
            print('Face detected = {}. Number of faces = {}'.format(faceDetected, num_faces))
            facefilename = '/'.join(facefilename.split('/')[-3:])
            print(facefilename)
            context = {
                'form': self.form,
                'num_faces': num_faces,
                'img_url': facefilename,
            }
            # print(to_send)

            # return HttpResponseRedirect('detect-faces')
            return render(request, 'textapp/image.html', context)

        else:
            print('Error: Form not valid')
            self.form = UploadFileForm()

            return render(request, 'textapp/image.html', {'form': self.form})


# def handle_uploaded_file(f):
#     with open('some/file/name.txt', 'wb+') as destination:
#         for chunk in f.chunks():
#             destination.write(chunk)


def upload_file(file):
    # file = request.files['image']
    print(file.size, file.name, file.content_type)

    # Save file
    facefilename = os.path.join(settings.BASE_DIR, 'textapp/static/textapp/img/', file.name)
    print(facefilename)
    # file.save(filename)

    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Detect faces
    faces = detect_faces(image)

    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)

        # Draw a rectangle
        for item in faces:
            draw_rectangle(image, item['rect'])

        # Save
        cv2.imwrite(facefilename, image)

        # In memory
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    # return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)
    return faceDetected, num_faces, to_send, facefilename


# ----------------------------------------------------------------------------------
# Detect faces using OpenCV
# ----------------------------------------------------------------------------------


def detect_faces(img):
    '''Detect face in an image'''

    faces_list = []

    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(os.path.join(settings.BASE_DIR, 'textapp/cascades/lbpcascade_frontalface.xml'))
    # Load OpenCV face detector (LBP is faster)
    face_cascade = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'textapp/cascades/haarcascade_frontalface_default.xml'))

    # Detect multiscale images (some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # If not face detected, return empty list
    if len(faces) == 0:
        return faces_list

    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {}
        face_dict['face'] = gray[y:y + w, x:x + h]
        face_dict['rect'] = faces[i]
        faces_list.append(face_dict)

    # Return the face image area and the face rectangle
    return faces_list
# ----------------------------------------------------------------------------------
# Draw rectangle on image
# according to given (x, y) coordinates and given width and heigh
# ----------------------------------------------------------------------------------


def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


# if __name__ == "__main__":
#     # Only for debugging while developing
#     app.run(host='0.0.0.0', debug=True, port=80)
