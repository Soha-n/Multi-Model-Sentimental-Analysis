import nltk
import re
import numpy as np
from statistics import mode
from nltk.classify import ClassifierI
import librosa
from pydub import AudioSegment
from keras.models import load_model
import pickle

from tensorflow.keras.preprocessing import image


stopwords_set = set(nltk.corpus.stopwords.words('english'))



class ImagePredictor:
    def __init__(self, model):
        self.model = model

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        
        img_array = image.img_to_array(img)
        
        img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def image_classify(self, image_path):
        
        processed_image = self.preprocess_image(image_path)

        predictions = self.model.predict(processed_image)

        predicted_class = np.argmax(predictions, axis=1)

        return str(predicted_class[0])



class AudioPredictor:
    def __init__(self, model):
        # Load the model directly
        self.model = model

    def preprocess_audio(self, audio_path):
        raw_audio = AudioSegment.from_file(audio_path)
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

    def audio_classify(self, audio_path):
        # Preprocess the audio file and extract features
        processed_audio, sr = self.preprocess_audio(audio_path)
        features = self.extract_features(processed_audio, sr)

        # Make predictions using the loaded model
        predictions = self.model.predict(features)

        # Get the predicted class
        predicted_class = np.argmax(predictions, axis=1)
        return str(predicted_class[0])

    def priority_fusion_audiotext(audio_emotion, text_emotion):
    
        if audio_emotion == text_emotion:
            return audio_emotion
    
    
        predictions = [audio_emotion, text_emotion]
        counter = Counter(predictions)
    
    
        final_emotion, count = counter.most_common(1)[0]
    
        if final_emotion != audio_emotion:
        
            return audio_emotion
    
        return final_emotion
        
    def extract_text_from_audio(audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
              audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

        

class VoteClassifier(ClassifierI):
    def __init__(self, vectorizer, *classifiers):
        self.vectorizer = vectorizer  # Add vectorizer to the class
        self._classifiers = classifiers

    def text_classify(self, sentence):
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

    def text_confidence(self, sentence):
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
