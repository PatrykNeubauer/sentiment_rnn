import numpy as np
import pandas as pd
import fasttext
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding
from keras.layers import Bidirectional
from keras.models import Model

embed_size = 300    # size of word vectors
max_features = 25000    # how many unique words to use
max_len = 100    # max number of word in text to use

# loading the dataset
dataset = pd.read_csv('dataset_formatted.csv', encoding='latin-1', header=None,
                      names=['sentiment', 'text'])

# separating labels and text
texts = dataset.text.values
labels = dataset.sentiment.values

# tokenization
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(texts))
texts = tokenizer.texts_to_sequences(texts)
texts = pad_sequences(texts, maxlen=max_len)

# acquiring the embedding matrix from word vectors
f = fasttext.load_model('cc.en.300.bin')
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = f.get_word_vector(word)
    embedding_matrix[i] = embedding_vector

# building the network
lstm_units_tries = [16, 32, 64, 128]
dropout_tries = [0.0, 0.2]
num_epochs = 10

for lstm_units in lstm_units_tries:
    for dropout in dropout_tries:
        inp = Input(shape=(max_len,))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
        x = Bidirectional(LSTM(lstm_units, dropout=dropout, return_sequences=True))(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=x)

        # training
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(texts, labels, batch_size=32, epochs=num_epochs, validation_split=0.2)

        # plots
        x_axis = range(num_epochs)
        plt.plot(x_axis, history.history['loss'], label='Loss (training data)')
        plt.plot(x_axis, history.history['val_loss'], label='Loss (validation data)')
        plt.title('Binary crossentropy for {} units, {} dropout'.format(str(lstm_units), str(dropout)))
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('loss_{}_{}.png'.format(str(lstm_units), str(dropout)))
        plt.close()

        plt.plot(x_axis, history.history['accuracy'], label='Accuracy (training data)')
        plt.plot(x_axis, history.history['val_accuracy'], label='Accuracy (validation data)')
        plt.title('Model accuracy for {} units, {} dropout'.format(str(lstm_units), str(dropout)))
        plt.ylabel('Accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.savefig('accuracy_{}_{}.png'.format(str(lstm_units), str(dropout)))
        plt.close()

