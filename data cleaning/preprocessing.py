import os
import pickle
import re
import string

import nltk
import pandas as pd
from definitions.hyper_parameters import MAX_LENGTH, RAND, VOCAB_SIZE
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

os.chdir("D:\Grad 2nd year\Thesis\Thesis Project")
df = pd.read_csv('final_data.csv')
df.rename(columns={"review_body": "text"}, inplace=True)

sentiment = [0 if i < 4 else 1 for i in df['star_rating']]  # set the sentiment conidion
df.insert(2, "sentiment", sentiment, True)

df_minority = df.loc[df['sentiment'] == 0]  # get the locations of negative reviews
df_majority = df.loc[df['sentiment'] == 1]  # location of positive reviews

# now we are using the locations from earlier to downsample
df_majority_downsampled = df_majority.sample(n=len(df_minority), replace=True, random_state=RAND)

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
# removing all rows where there is a non-string in the text (there is only 1)

df_downsampled = df_downsampled[df_downsampled['text'].map(type) == str]

df_downsampled = df_downsampled[['sentiment', 'text']]


# A function to replace contractions that come of from the tokenization

def decontracted(phrase):
    # specific
    # general
    phrase = re.sub(r"n\'t", "not", phrase)
    phrase = re.sub(r"\'re", "are", phrase)
    phrase = re.sub(r"\'s", "is", phrase)
    phrase = re.sub(r"\'d", "would", phrase)
    phrase = re.sub(r"\'ll", "will", phrase)
    phrase = re.sub(r"\'t", "not", phrase)
    phrase = re.sub(r"\'ve", "have", phrase)
    phrase = re.sub(r"\'m", "am", phrase)
    return phrase


def clean(df):
    sentence = []  # we will put the string into this list
    for i in range(len(df)):  # go down the dataframe
        tokens = nltk.word_tokenize(df.iat[i, 1])  # go through each word in the string and make it a token
        tokens = [decontracted(word) for word in tokens]  # replace contractions from the tokens
        cleaned_tokens = ""
        for token, tag in pos_tag(tokens):
            # remove punctuation from each token
            table = str.maketrans('', '', string.punctuation)
            tokens = [word.translate(table) for word in tokens]
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            # filter out short tokens
            tokens = [word for word in tokens if len(word) > 1]
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower():
                cleaned_tokens = cleaned_tokens + " " + token.lower()
        sentence.append(cleaned_tokens)
    # integer encode the documents
    encoded_docs = [one_hot(word, VOCAB_SIZE) for word in sentence]
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_LENGTH, padding='post')
    return padded_docs


reviews = clean(df_downsampled)
df_downsampled.to_csv('sentiment.csv')  # saving the data with the review text and sentiment as a csv
with open('reviews', 'wb') as f: pickle.dump(reviews, f)  # saving the reviews as a pickle
