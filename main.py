import nltk

import string
import random

with open('./data/sample.txt', 'r', errors='ignore') as file:
    raw = file.read().lower()

# Download the necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

# Tokenize the text into sentences using nltk's sent_tokenize
sent_tokens = nltk.sent_tokenize(raw)

# Tokenize the text into words using nltk's word_tokenize
word_tokens = nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "namaste", "hy")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
# vectorization
        
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    text_pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(tokenizer=LemNormalize, stop_words='english')),
    ('tfidf_transformer', TfidfTransformer())
])
    tfidf = text_pipeline.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatbot_response = chatbot_response+"I am sorry! I don't understand you."
        return chatbot_response
    else:
        chatbot_response = chatbot_response+sent_tokens[idx]
        return chatbot_response

flag = True
print("Chatbot: My name is Chatbot. I will answer your queries about Machine Learning. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print('Chatbot: You are welcome.')
        else:
            if(greeting(user_response) != None):
                print("Chatbot: "+ greeting(user_response))
            else:
                print("Chatbot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Chatbot: Bye! take care.")


        
