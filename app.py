from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re


app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])

def home():
    def cleansing(data):
        # lower text
        data = data.lower()
    
        # hapus punctuation
        remove = string.punctuation
        translator = str.maketrans(remove, ' '*len(remove))
        data = data.translate(translator)
        
        # remove ASCII dan unicode
        data = data.encode('ascii', 'ignore').decode('utf-8')
        data = re.sub(r'[^\x00-\x7f]',r'', data)
        
        # remove newline
        data = data.replace('\n', ' ')
    
        return data

    

    #data di dapatkan dari web scrapping 
    df = pd.read_csv("data\data_coffeeshop.csv")
    #split dataset
    from sklearn.model_selection import train_test_split

    X = df['Review text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, stratify=y, random_state=30)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    def preprocess_data(data):
        #cleansing data
        data = cleansing(data)
    
        # hapus stopwords
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        data = stopword.remove(data)
        
        # stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        data = stemmer.stem(data)
        
        # count vectorizer
        
        data = vectorizer.transform([data])
        return data

    from sklearn import svm
    from sklearn.model_selection import cross_val_score

    clf = svm.SVC(kernel="linear")
    clf.fit(X_train,y_train)
    predict = clf.predict(X_test)

    
    data1 = request.form['sentiment']
    # return data1
   
    pred = clf.predict(preprocess_data(data1))
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
