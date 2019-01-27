
import nltk
from nltk import *
import pickle
from flask import Flask,render_template,url_for,request
from sklearn_crfsuite import CRF
import json


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST'])
def predict():
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features
    
    if request.method == 'POST':
        message = request.form['message']

        crf_1 = pickle.load(open('finalized_model.sav', 'rb'))
        text = word_tokenize(message)
        sentence = list(nltk.pos_tag(text))
        sentence_features = [word2features(sentence, index) for index in range(len(sentence))]
        #print(list(zip(sentence, crf_1.predict([sentence_features])[0])))
        dict_={}
        m=list(zip(sentence, crf_1.predict([sentence_features])[0]))
        for i in range(0,len(m)):
            if m[i][1] not in dict_.keys():
                dict_[m[i][1]]=[m[i][0][0]]
            else:
                dict_[m[i][1]].append(m[i][0][0])
        json_result = json.dumps(dict_)
    return render_template('result.html',prediction =json_result )


if __name__ == '__main__':
    app.run(debug=True)