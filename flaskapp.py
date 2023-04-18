from flask import Flask, render_template
import pandas as pd
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from flask import request

app = Flask(__name__)

all_feature_descr = pd.read_excel('Zoom-features-2022.xlsx', sheet_name = None, usecols = [2]) # read second column from all sheets, returns dictionary {sheet_name:dataframe}
months = ['Jan-2022','Feb-2022','March-2022','April-2022','May-2022','June-2022','July-2022','Aug-2022','Sept-2022','Oct-2022','Nov-2022','Dec-2022']
num_distinguishers = 5
this_month = 3

# download nltk utilities
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Text preprocessing code, preprocess(string) function should be run on each feature details to perform stripping, stemming, lemmatization, etc.
#convert to lowercase, strip and remove punctuations
def strip(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

# STOPWORD REMOVAL and remove the word "zoom"
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english') and i != 'zoom']
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
        
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

# performs all preprocessing on a given string
def preprocess(string):
  return lemmatizer(stopword(strip(string)))

# print("\nFeatures Preprocessed: ")

# for value in feature_descriptions_dec_22.values:
#   print(preprocess(value[0]))


# text vectorization code to create a vocabulary (bag of words)
cv = CountVectorizer()

def tokenization(feature_descr, sentences): # We tokenize the data here by calling Jackson's preprocess function featured above.
  for month in months:
    for value in feature_descr.get(month).values:
      if isinstance(value[0], str) and len(value[0]) > 0:
        sentences.append(tokenize.sent_tokenize(preprocess(value[0])))
  return oneArray([], sentences)

def oneArray(arrayData, sentences): # Takes the array of arrays and turns it into one single array so that the cv functions can work properly. Data needed to be in one array.
  num = 0
  for i in sentences:
    arrayData += sentences[num:][0]
    num += 1
  return cvFit([], arrayData)

def cvFit(vectorizedData, arrayData): # cv.fit takes the data and separates each unique word, while also giving it a unique identifier, being the number shown beside the word.
  vectorizedData = cv.fit(arrayData)
  #print("Every unique word and their given unique identifier:")
  #print(vectorizedData.vocabulary_) #lists out each unique word with a unique identifier
  #print("-"*50)
  #print("Every unique word in array format:")
  #print(vectorizedData.get_feature_names_out()) #lists out each unique word
  #print("-"*50)
  return cvTransform(vectorizedData, arrayData)

def cvTransform(vectorizedData, arrayData): # cv.transform takes the array and vectorizes the data
  vectorizedData = cv.transform(arrayData)
  #print("Number of sentences(left) and number of unique words(right)")
  #print(vectorizedData.shape) # shows the total number of sentences as the first number and the second number represents the number of unique words from all of those sentences.
  #print("-"*50)
  #print("Array that represents the frequency of each unique word in each sentence, each array within the array represents a sentence:")
  #print(vectorizedData.toarray())# This shows the array of all unique words by their unique idenfitifier, so the first entry in the array is the word 'ability' with the idenitifer '0', 
                                # and the number in the array itself represents the frequency of that word occurring in each sentence. Each array represents a sentence.
  #print("-"*50)
  #print("Sentence number(left), unique word identifier(right) and frequency of given word in sentence (far right):")
  #print(vectorizedData) # The first number represents the sentence number, and the second number represents the unique word. The number to the right is the frequency of the word
  return vectorizedData.toarray()

def vocab(feature_descr, sentences): # We tokenize the data here by calling Jackson's preprocess function featured above.
  for month in months:
    for value in feature_descr.get(month).values:
      if isinstance(value[0], str) and len(value[0]) > 0:
        sentences.append(tokenize.sent_tokenize(preprocess(value[0])))
  return singleArray([], sentences)

def singleArray(arrayData, sentences): # Takes the array of arrays and turns it into one single array so that the cv functions can work properly. Data needed to be in one array.
  num = 0
  for i in sentences:
    arrayData += sentences[num:][0]
    num += 1
  return vocabOutput([], arrayData)

def vocabOutput(vectorizedData, arrayData): # cv.fit takes the data and separates each unique word, while also giving it a unique identifier, being the number shown beside the word.
  vectorizedData = cv.fit(arrayData)
  return vectorizedData.vocabulary_

# vocab_sentences = []
# vocab(vocab_sentences)

# print(vocab_sentences)

# supervised learning code to train a model

# Provide the date that distinguishes old and new, and the desired number of distinguishers
def supervisedLearning(feature_descr, numOfDistinguishers,divider):
  data = []
  feature_set = []
  # Define the binary feature matrix and the feature names
  x = []
  x = tokenization(feature_descr, data)
  features = vocab(feature_descr, feature_set)
  # Define the binary labels for old (0) and new (1) requirements using date seperator
  y = []
  month_counter = 0
  for month in months:
    month_counter += 1
    for value in feature_descr.get(month).values:
      if isinstance(value[0], str) and len(value[0]) > 0:
        if(month_counter <= divider):
          y.append(0)
        else:
          y.append(1)

  # Fit a logistic regression model to the data
  lr = LogisticRegression(solver='liblinear')
  lr.fit(x, y)

  # Print the top coefficients for old and new features
  coefs = lr.coef_[0]
  top_old_indices = coefs.argsort()[:numOfDistinguishers]
  top_new_indices = coefs.argsort()[-numOfDistinguishers:][::-1]
  print("Distinguishers of old features:")
  top_old_words = []
  for idx in top_old_indices:
    for key in features:
      if features[key] == idx:
        print(key)
        top_old_words.append(key)
  print("\nDistinguishers of new features:")
  top_new_words = []
  for idx in top_new_indices:
    for key in features:
      if features[key] == idx:
        print(key)
        top_new_words.append(key)

  # Split the data into training and test sets
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9)

  # Fit a logistic regression model to the training data
  lr = LogisticRegression(solver='liblinear')
  lr.fit(x_train, y_train)

  # Select the top distinguishers for old and new requirements
  coefs = lr.coef_[0]
  top_old_indices = coefs.argsort()[:numOfDistinguishers]
  top_new_indices = coefs.argsort()[-numOfDistinguishers:][::-1]
  top_indices = np.concatenate([top_old_indices, top_new_indices])

  # Generate a new feature matrix with only the top distinguishers
  x_train_new = x_train[:, top_indices]
  x_test_new = x_test[:, top_indices]

  # Fit a logistic regression model to the new training data
  lr_new = LogisticRegression(solver='liblinear')
  lr_new.fit(x_train_new, y_train)

  # Make predictions on the test set
  y_pred = lr_new.predict(x_test_new)

  # Compute the accuracy, precision, and recall
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  print()
  print("Divide: " , divider, " months old features, " , (12-divider), "months new features.") 
  print("Accuracy: " , round(accuracy*100,2) , "%")
  # Can do precision and recall if needed, future research needed for optimization
  # print("Precision: " , round(precision*100,2) , "%")
  # print("Recall: " , round(recall*100,2), "%")

  # return top_old_words, top_new_words, accuracy, precision, recall
  return top_old_words, top_new_words, accuracy, precision, recall

  
#supervisedLearning(20,3)

# home page
@app.route("/")
def home():
  global num_distinguishers
  global this_month
  num_distinguishers = int(request.args.get('numDistinguishers', 5))
  this_month = int(request.args.get('month', 3))
  return render_template('home.html')

# results page, feature_set_id is the id of the feature set to be used (zoom or webex, maybe custom in the future)
@app.route("/results/<feature_set_id>")
def results(feature_set_id = "zoom"):
  global num_distinguishers
  global this_month
  feature_descr = pd.read_excel('Zoom-features-2022.xlsx', sheet_name = None, usecols = [2]) # read second column from all sheets, returns dictionary {sheet_name:dataframe}
  # run supervised learning code and return the result
  if(feature_set_id == "zoom"):
    # update feature_descr to zoom
    feature_descr = pd.read_excel('Zoom-features-2022.xlsx', sheet_name = None, usecols = [2]) # read second column from all sheets, returns dictionary {sheet_name:dataframe}
    top_old_words, top_new_words, accuracy, precision, recall = supervisedLearning(feature_descr, num_distinguishers,this_month)
    # maybe we can actually run this multiple times (for month cutoffs) and return the best result and what month cutoff it was
  
  elif(feature_set_id == "webex"):
    # update feature_descr to webex
    feature_descr = pd.read_excel('WebEx-features-2022.xlsx', sheet_name = None, usecols = [2]) # read second column from all sheets, returns dictionary {sheet_name:dataframe}
    # run supervised learning code and return the result
    top_old_words, top_new_words, accuracy, precision, recall = supervisedLearning(feature_descr, num_distinguishers,this_month)

  return render_template('result.html', top_old_words=top_old_words, top_new_words=top_new_words, accuracy=accuracy, precision=precision, recall=recall)

