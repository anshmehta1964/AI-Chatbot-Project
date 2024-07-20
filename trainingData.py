import random
import json
import pickle
import numpy as np


#natural languvage tool kit
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!',',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        # print(word_list)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        # at the beginning there will be empty classes list so we need to fill it
        # we will do it such as below so each unique tags will be appended in classes list based on which output will be produced
        # print("classes: ",classes)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# if the word is not in ignore_words than put in inside words and lemmatize it.
# Example : Greetings - greeting
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]


# using set so duplicates are removed and sorting them
# Note :- you might think that the output may vary due to sorting but it wont as the output is bases on pair of words.
words = sorted(set(words))
classes = sorted(set(classes))
# print(words)
print(len(documents),"Documents: ",documents)
print(len(classes)," Classes: ",classes)
print(len(words)," Unique Lemmatized words: ",words)

# Creating the pickle files which will be written in binary
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
# The number of zeroes will be total number of classes which in our case is 18
output_empty = [0] * len(classes)

# This is the most important part
# what is documents? documents means each pair of a patter with tag which in our case is 58(Number of patterns) docs.
# In simple words tag is paired with each pattern which equals to Documents = patterns(with tags)
for document in documents:
    # Initializing bag of words
    bag =[]
    # List of tokenized words for the pattern
    # why document[0]? because in document we paired (lemmatize_words,intent['tags']) WE DONT WANT TAGS WE JUST WANT WORDS that's why documents[0]
    word_patterns = document[0]
    print(word_patterns)

    # Lemmatizing each pattern and storing it in the word_patterns
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    print("Lemmatized: ",word_patterns)
    print(len(words))
    for w in words:
        # it will put 1 only there if the word exists in that pattern else 0
        bag.append(1) if w in word_patterns else bag.append(0)
        # In short if the word_patterns word are matched with words[list] than it will put 1 on that places simple
    print("Current word is: ", word_patterns)
    print("Current Bag is: ", bag)
    print(len(bag))

    # Output will be 0 for each tag and 1 for current tag(for each pattern)
    output_row = list(output_empty)
    # we will get the classes then put indexes accordingly
    #print("Class index",classes.index(document[1]))
    output_row[classes.index(document[1])] = 1
    #print("output row is",output_row)
    training.append([bag, output_row])

    # final training will look like
    # ([0,1,0,0,0,0........1,0],[1])

# It might be useful to state exactly why the answer in the first link didn't help you.
# Otherwise, we're taking the risk of repeating content already said there with little improvements.
random.shuffle(training)
# tensorflow only takes np so we have to give him array of np
training = np.array(training)

# print("training: ",training)
train_x = list(training[:, 0])
# print("train_x", train_x) - for words
train_y = list(training[:, 1])
# print("train_y", train_y) - for classes


#-------------------------------------------------ANN----------------------------------------------------#
# The input units(receptor), connection weights, summing function, computation and output units (effectors) are what makes up an artificial neuron.
model = Sequential()
# Input shape is the number of columns present in the data set
# The output node can have different activation functions and different functions have different functionalities.
# Following are the functions Such as Linear{-infinity, +infinity}, Heviside,step func{0,1},Sigmoid func, Signum{-1,1}
#Rectified Linear Unit {0, +infinity}
# Leaky ReLU function {-infinity, +infinity}
# x[0] for words
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Dropout simply means removing some neurons from the layers but why to drop neurons? Because if you drop some neurons than the model
# will try to increase its accuracy
# Dropout helps over come the problem of overfitting
# Overfitting - a model is certainly so good on classifying data that it was trained on but not so good in classifying data that it wasn't trained on.
model.add(Dropout(0.8))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# y[0] for classes
# softmax = exponential of particular class / sum of exponential of each class
model.add(Dense(len(train_y[0]), activation='softmax'))

# SGD is the optimizer we used
# Optimizer updates the model in response to the output of the loss function
# Optimizers assist in minimizing the loss function
# Learning rate means the steps, the graph will go down on basis of the loss value and each time it will have a lesser loss value
# the plot on the graph where it's getting lower is called learning rate
# If the learning rate is very high you will reach the minimum so soon but you won't be able to get the true minimum value
# if the learning rate is low then it will take more epochs but you will be able to get a good minimum loss value and you might also get a false minimum
# Momentum basically speed ups the process it will help getting the minima

# Decay means when you take steps towards the minimum how much smaller steps you want to take, it keeps getting small as you keep getting close
# Nesterov is a math function it is also known as look ahead term
# Nesterov Accelerated Gradient is a momentum-based SGD optimizer that "looks ahead" to where the parameters will be to calculate the gradient ex post rather than ex ante
#Stochastic gradient descent (often abbreviated SGD) is an iterative method for optimizing an objective function with suitable smoothness properties
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Before training the model we need to compile it and define the loss function, optimizers, and metrics for prediction. We compile the model using . compile() method.

#Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right.
# Formally, accuracy has the following definition: Accuracy = Number of correct predictions Total number of predictions.
# cross entrpohy = -(1*log(high_probability)+0*log(low_probability)) - also known as hot encoding
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Keras fit is the method used for the model training on the data set for the specified number of fixed epochs or iterations mentioned.
#The batch size is a number of samples processed before the model is updated.
# Verbose is use to check overfitting
# Epochs means repeating the same data 200 times.
# Batch size means at each epoch the segment to data is learned, why segment? in case if you have 10 millions plots it will take time for computer
# to remember 10M plots thus it's easy to give it in batch with specified size
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=500, verbose=1)
model.save('chatbotmodel2.h5', hist)

print('Done')