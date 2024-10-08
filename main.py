import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import load_model

##load data sets word index
word_index=imdb.get_word_index()
reverse_word_index = {value:key for key,  value in word_index.items()}

##load trained model with Relu activation
model = load_model('simple_rnn_imdb.h5')


## decode reviews
def decode_reviews(encoded_reviews):
    return ' '.join([reverse_word_index.get(i -3,'?') for i in encoded_review])

##preprocessing user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

## prediction function
def prediction_sentiment(review):
    processed_text = preprocess_text(review)
    prediction = model.predict(processed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as Positive or Negative")

##user input
user_input = st.text_area('Movie review')

if st.button('Classify'):
    processed_text = preprocess_text(user_input)

    ##make prediction
    prediction = model.predict(processed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'



    ##display Result
    st.write(f'sentiment is {sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
        st.write('Please enter a review')






