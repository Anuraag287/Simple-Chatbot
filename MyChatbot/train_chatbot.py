import nltk
nltk.download('punkt')
nltk.download('wordnet')
import tensorflow as tf
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import numpy as np
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes=[]
documents=[]
ignore=['!','?']
data=open('intents.json').read()
intents=json.loads(data)
for intent in intents['intents']:
    for pat in intent['patterns']:
        tokword=nltk.word_tokenize(pat)
        print(tokword)
        words.extend(tokword)
        documents.append((tokword,intent['tag']))
        if(intent['tag'] not in classes):
            classes.append(intent['tag'])
print(words)
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
#print(len(documents))
#print(documents)
#print(classes)
pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))

training=[]
output=[0]*len(classes)
for doc in documents:
    bag=[]
    pattwords=doc[0]
    pattwords=[lemmatizer.lemmatize(word.lower())for word in pattwords]
    for w in words:
        bag.append(1) if w in pattwords else bag.append(0)
    output_row=list(output) 
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
print("training data created")
#print(train_x)
#print(train_y)
#print(pattwords)
#print(output_row)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")

