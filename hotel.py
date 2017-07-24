import pandas as pd
# from pip._vendor.pyparsing import delimitedList
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn import svm
# from sklearn.metrics import accuracy_score
from nltk import sent_tokenize, word_tokenize, pos_tag
# import matplotlib.pyplot as plt
# from collections import Counter
from nltk.corpus import stopwords
import re, string
import os
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.naive_bayes import GaussianNB


classes = ["Sentiment",'Food', 'Service', 'Value', 'Ambience', 'Cleanliness', 'Amenities', 'Hospitality', 'Location', 'Front-Desk', 'Room']


lem = WordNetLemmatizer()

path1 = r"/home/repusight/Desktop/googlereview_sarovar_21apr17.csv"

data = pd.read_csv("train_hotel.csv")
test = pd.read_csv(os.path.normpath(path1),delimiter="|")

#train, test = train_test_split(data, test_size = 0.2)

# data_text = data['Comment']
#trainData = data[['Comment', 'Sentiment']].copy()

#test_label = test["Sentiment"]


wordDict = {"can't": "can not", "don't": "do not", "i'll": "i will", "it's": "it is", "didn't": "did not",
            "could'nt": "could not", "you'll": "you will", "wasn't": "was not", "won't": "would not",
            "I'm": "I am"}


def remove_punctuation(input_text):
    input_text = re.sub(r'[^\w\s]','',input_text)
    return input_text


noise_list = set(stopwords.words('english'))


def remove_noise(input_text):
    words = word_tokenize(input_text)
    noise_free_words = [word for word in words if word not in noise_list]
    noise_free_text = " ".join(noise_free_words)
    return noise_free_text


def sentence_tokenizer(text):
    sentences = sent_tokenize(text)
    return sentences


def word_tokenizer(text):
    words = word_tokenize(text)
    return words


def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text


def cleanedComment(text):
    text = text.lower()
    text = multipleReplace(text, wordDict)
    text = remove_punctuation(text)
    text = remove_noise(text)
    return text


def sentiment_process():
    train_cleanedData = []
    for txt in data.iterrows():
        train_cleanedData.append(cleanedComment(txt[1]['Comment']))
    vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3),stop_words = 'english')
    train_vectors = vectorizer.fit_transform(train_cleanedData)
    vocab = vectorizer.get_feature_names()
    print(len(vocab))

    test_cleanedData = []
    for txt in test.iterrows():
        #print(txt)
        test_cleanedData.append(cleanedComment(txt[1]['Comment']))
    test_vectors = vectorizer.transform(test_cleanedData)

    train_vectors = train_vectors.toarray()
    test_vectors = test_vectors.toarray()

    for cls in classes:
        print("Before Model")
        model = svm.SVC(kernel='linear')
        print(cls)
        #model = GaussianNB()
        model.fit(train_vectors, data[cls])
        print("After Model")
        prediction = model.predict(test_vectors)
        print("After Prediction")
        test[cls] = prediction
        #print(test)
    test.to_csv("/home/repusight/Desktop/googlereview_sarovar_21apr17_classified.csv", '|', encoding='utf-8', index=False, header=True)
    # print(classification_report(test_label, prediction))
    # print(confusion_matrix(test_label, prediction))


sentiment_process()

