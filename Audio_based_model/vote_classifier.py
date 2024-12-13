import nltk
import re
import numpy as np
from statistics import mode
from nltk.classify import ClassifierI
import librosa
from pydub import AudioSegment
from keras.models import load_model
import pickle

stopwords_set = set(nltk.corpus.stopwords.words('english'))

import numpy as np


class AudioPredictor:
    def __init__(self, model_path, emotion_labels):
        # Load the audio model
        self.model = load_model(model_path)
        self.emotion_labels = emotion_labels

    def preprocess_audio(self, path):
        raw_audio = AudioSegment.from_file(path)
        samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
        
        # Trim silence from the beginning and end
        trimmed, _ = librosa.effects.trim(samples, top_db=25)

        # Pad to a fixed length if necessary (e.g., 180,000 samples)
        padded = np.pad(trimmed, (0, max(0, 180000 - len(trimmed))), 'constant')
        sr = raw_audio.frame_rate  # Get the sample rate

        return padded, sr

    def extract_features(self, audio, sr):
        # Extract features similar to training
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Concatenate features to create the input array
        features = np.concatenate((
            np.swapaxes(zcr, 0, 1), 
            np.swapaxes(rms, 0, 1), 
            np.swapaxes(mfccs, 0, 1)), 
            axis=1
        )

        # Reshape for LSTM input (1 sample, timesteps, features)
        features = features.reshape(1, features.shape[0], features.shape[1])  # Add batch dimension
        return features

    def predict(self, audio_path):
        # Preprocess the audio file and extract features
        processed_audio, sr = self.preprocess_audio(audio_path)
        features = self.extract_features(processed_audio, sr)

        # Make predictions using the loaded model
        predictions = self.model.predict(features)

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)
        return self.emotion_labels[predicted_class[0]]



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
