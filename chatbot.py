import random
import json
import pickle
import numpy as np
import nltk
import webbrowser   
import os

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from tensorflow.keras.models import load_model

model = load_model('model.h5')

intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    return classes[np.argmax(res)]

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

print("Bot is running! Type 'quit' to exit")

import webbrowser
import os

while True:
    message = input("You: ")

    if message.lower() == "quit":
        break

    tag = predict_class(message)
    response = get_response(tag)

    print("Bot:", response)

    if "scholarship" in message.lower():
        webbrowser.open("https://ssp.karnataka.gov.in/")

    # OPEN PLACEMENT PAGE
    if "open placement" in message.lower():
        path = os.path.abspath("placement.html")
        webbrowser.open("file://" + path)

    # OPEN LIBRARY PAGE
    if "open library" in message.lower():
        path = os.path.abspath("library.html")
        webbrowser.open("file://" + path)

    # OPEN FEES PAGE
    if tag == "fees":
        path = os.path.abspath("fees.html")
        webbrowser.open("file://" + path)

    # OPEN UG CUTOFF PDF
    if "cutoff" in message.lower() or "cut-off" in message.lower():
        path = os.path.abspath("SJBIT_UG_Cutoff_Expanded.pdf")
        webbrowser.open("file://" + path)

    # OPEN MTECH PDF
    if "mtech" in message.lower() or "pg cutoff" in message.lower():
        path = os.path.abspath("SJBIT_PG_Cutoff_MTech.pdf")
        webbrowser.open("file://" + path)

    # open hostel only when user asks
    if message.lower() == "open hostel":
        path = os.path.abspath("hostel.html")
        webbrowser.open("file://" + path)