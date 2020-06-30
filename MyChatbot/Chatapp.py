import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')
def clean_sentences(sentence):
    tokword=nltk.word_tokenize(sentence)
    tokword=[lemmatizer.lemmatize(w.lower()) for w in tokword]
    return tokword

def bow(sentence,word,show_details=True):
    bag=[0]*len(word)
    sentences=clean_sentences(sentence)
    for s in sentences:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1 
                if (show_details):
                    print("Word found")
    return (np.array(bag))

def predict_class(sentence,model):
    p=bow(sentence,words,show_details=False)
    res=model.predict(np.array([p]))[0]           
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res
    