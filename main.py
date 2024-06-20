import os
import pickle
import json
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


with open("dataset.json") as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for patter in intent['patterns']:
            wrds = nltk.word_tokenize(patter.lower())
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if (intent['tag'] not in labels):
            labels.append(intent['tag'])

    words = [lemmatizer.lemmatize(w) for w in words]
    words = [stemmer.stem(w) for w in words if w != '?']
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for idx, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[idx])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)


if not os.path.isfile('model.keras'):
    input_shape = len(training[0])

    X_train, X_test, y_train, y_test = train_test_split(
        training, output, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(
            input_shape,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(len(output[0]), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=1000, batch_size=8,
              validation_data=(X_test, y_test))

    model.save('model.keras')
else:
    model = keras.models.load_model('model.keras')


def bag_of_words(user_input, words):
    bag = [0 for _ in range(len(words))]
    user_input_words = nltk.word_tokenize(user_input.lower())
    user_input_words = [stemmer.stem(w.lower()) for w in user_input_words]
    for se in user_input_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat():
    print('Start talking with the bot (type "quit" to stop)!')

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break

        input_bag = bag_of_words(user_input, words)
        result = model.predict(np.array([input_bag]))

        # Get the index of the highest probability
        result_index = np.argmax(result)
        tag = labels[result_index]

        for intent in data['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
                response = random.choice(responses)
        print('RestaurantBot:', response)


chat()
