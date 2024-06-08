import streamlit as st
import numpy as np

# Potential update for TensorFlow version
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense





# Sample data
texts = ["I love this movie", "I hate this movie", "This movie is okay"]
labels = np.array([0, 1, 2])  # 0 - positive, 1 - negative, 2 - neutral

# Tokenize the text
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform length
maxlen = 10
data = pad_sequences(sequences, maxlen=maxlen)


# Define the model architecture for RNN
model_rnn = Sequential()
model_rnn.add(Embedding(max_words, 64))
model_rnn.add(SimpleRNN(32))
model_rnn.add(Dense(3, activation='softmax'))
model_rnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_rnn.fit(data, labels, epochs=5, batch_size=1, verbose=0)


# Define the model architecture for LSTM
model_lstm = Sequential()
model_lstm.add(Embedding(max_words, 64))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(3, activation='softmax'))
model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(data, labels, epochs=5, batch_size=1, verbose=0)


# Define the model architecture for GRU
model_gru = Sequential()
model_gru.add(Embedding(max_words, 64))
model_gru.add(GRU(32))
model_gru.add(Dense(3, activation='softmax'))
model_gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_gru.fit(data, labels, epochs=5, batch_size=1, verbose=0)





# Streamlit UI
st.title("Sentiment Analysis with RNN, LSTM, and GRU")

text_input = st.text_input("Enter your text:")

selected_model = st.selectbox("Select Model", ["RNN", "LSTM", "GRU"])

if st.button("Predict"):
    # Preprocess input text
    test_texts = [text_input]
    test_sequences = tokenizer.texts_to_sequences(test_texts)
    test_data = pad_sequences(test_sequences, maxlen=maxlen)

    if selected_model == "RNN":
        prediction = model_rnn.predict(test_data)
    elif selected_model == "LSTM":
        prediction = model_lstm.predict(test_data)
    elif selected_model == "GRU":
        prediction = model_gru.predict(test_data)

    sentiment = np.argmax(prediction)

    # Emoticons
    if sentiment == 0:
        emoticon = "😍"  # Positive sentiment
    elif sentiment == 1:
        emoticon = "😡"  # Negative sentiment
    else:
        emoticon = "😐"  # Neutral sentiment

    st.write(f"{selected_model} Model Prediction:", emoticon)
    st.write("----")