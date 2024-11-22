import nltk
import re
import numpy as np
from statistics import mode
from nltk.classify import ClassifierI
import pickle

stopwords_set = set(nltk.corpus.stopwords.words('english'))

class VoteClassifier(ClassifierI):
    def __init__(self, vectorizer, *classifiers):
        self.vectorizer = vectorizer  # Add vectorizer to the class
        self._classifiers = classifiers

    def classify(self, sentence):
        votes = []
        features_tfidf = self.vectorizer.transform([sentence])
        processed_features = self.preprocess_nltk(sentence)

        for clf in self._classifiers:
            if hasattr(clf, 'predict'):
                prediction = clf.predict(features_tfidf)
                votes.append(str(prediction[0]))
            elif hasattr(clf, 'classify'):
                prediction = clf.classify(processed_features)
                votes.append(str(prediction))
            else:
                cleaned_sentence = self.clean_new_text(sentence)
                prediction = clf.predict(np.array([cleaned_sentence]))
                predicted_class = np.argmax(prediction, axis=1)[0]
                votes.append(str(predicted_class))

        return mode(votes)

    def preprocess_nltk(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha() and token not in stopwords_set]
        return {word: True for word in tokens}

    def confidence(self, sentence):
        votes = []
        features_tfidf = self.vectorizer.transform([sentence])
        processed_features = self.preprocess_nltk(sentence)

        for clf in self._classifiers:
            if hasattr(clf, 'predict'):
                prediction = clf.predict(features_tfidf)
                votes.append(str(prediction[0]))
            elif hasattr(clf, 'classify'):
                prediction = clf.classify(processed_features)
                votes.append(str(prediction))

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

    def clean_new_text(self, text):
        stemmer = nltk.stem.PorterStemmer()
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower()
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stopwords_set]
        return " ".join(words)
