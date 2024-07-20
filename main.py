import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Reading intents
intents = json.loads(open('intents.json').read())

# Reading words, Classes and loading the model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    # Tokenize words will return a list of words converted from a paragraph,line etc.
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatizer will lemmatize the words and only store the meaning full words in sentence words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    # Creating bag of words initialize bag with 0 but how many? Answer is total no. of words
    bag = [0] * len(words)
    # print(words)
    for w in sentence_words:
        # Now we will compare the words with pickle file words if the words are present in the pickle file we put 1 else 0 in Bow
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    # in the array where it will find the word corresponding to the user entered word it will put one there
    # print(np.array(bag))
    return np.array(bag)


def predict_class(sentence):
    # Getting Numpy array in return from bag_of_words function
    bow = bag_of_words(sentence)
    # Passing the array as list in predict function and 0th index
    # 0'th index is just to match the format
    # k = np.array([bow])
    # print(k)
    res = model.predict(np.array([bow]))[0]
    # print(res)
    # Error threshold means the uncertainty , if the uncertainty is high then don't take that result 0.25 is 25%
    ERROR_THRESHOLD = 0.25
    # i means total number of classes and r means the chances or we can say that the word existed in the class after process of optimization
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        # finding the classes in which the word existed
        # print(classes[r[0]],str(r[1]))
        # print(return_list)
    return return_list


def get_response(intents_list, intents_json):
    # Now we are getting the tag with highest probability 0th index contains tag name and 1 index contains the probability percentage
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        # traverse all the tags and if the tag is similar to the tag we got with highest probability then select any random answer from that tag patterns
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


while True:
    message = input("| You: ")
    if message == "bye" or message == "Goodbye" or message =="goodbye":
        # ints = predict_class(message)
        # res = get_response(ints, intents)
        print("| Bot:")
        print("Hope you like the bot Created by Ansh & yash, cya next time!")
        exit()

    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
