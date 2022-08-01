from typing_extensions import final
import joblib
import pandas as pd # pandas package

import numpy as np # numpy package

# matplotlib packages
import matplotlib
import matplotlib.pyplot as plt 
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.figsize'] = (36.0, 20.0)

import seaborn as sns # seaborn package

# plotly packages
from chart_studio import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot


import warnings  # warnings package
warnings.filterwarnings('ignore')

# dictionary package
from collections import Counter, defaultdict

import re #regex package
from textblob import TextBlob #import textblob package

# word cloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# nltk packages
import nltk

#nltk.download('stopwords')
# stop words
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))

# detokenizer 
from nltk.tokenize.treebank import TreebankWordDetokenizer

# punctuation
from string import punctuation


# pickle package
import joblib

#stream lit
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# ------------------------------------------------------

# punctuation dictionary
punctuation = set(punctuation) 
include_punctuation = {'’', '”', '“'}
punctuation |= include_punctuation

# stop words and other words to be excluded
include_stopwords = {'could', 'shouldn', 'oh', 'know', 'im', 'en',
'go', 'get', 'got', 'gonna', 'la', 'na', 'de', 'gon', 'got' 'must', 'would', 'also', 
                    'apple', 'Apple', 'Amazon', 'amazon', 
                     'roku', 'Roku', 'roku remote', 'Rokue Remote',
                     'Google', 'google', 'chromecast', 'Chromecast', 
                    'Chrome Cast', 'chrome cast', 'chrome', 'cast'
                     'Fire TV Stick', 'prime', 'firestick4ktv',
                     'firestick', 'fire tv', 'fire tv stick', 'fire', 
                     'firesticks','tv', 'remote', '4k', 'stick', 'dont', "it's", 'tvs',
                    'etc'}

# include the dictionary of stop words
sw |= include_stopwords

# useful white space pattern
whitespace_pattern = re.compile(r"\s+")

def decontracted(phrase):
    """
    split up decontracted words from a column of texts
    
    """
    # add extra white space
    phrase = re.sub('(?<=[.,!?()/:;])(?=[^\s])', r' ',  phrase)
   
    # specific
    phrase = re.sub(r"she/her", "she her",phrase)
    phrase = re.sub(r"he/him", "he him",phrase)
    phrase = re.sub(r"they/them", "they them",phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r'\<.*\>', '', phrase)

    # general
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"don’t", "do not", phrase)
    phrase = re.sub(r"it's", "it is", phrase)
    phrase = re.sub(r"it’s", "it is", phrase)
    phrase = re.sub(r"we've", "we have", phrase)
    phrase = re.sub("\w+\d+", "", phrase)
    phrase = re.sub("\d+\w+", "", phrase)
    phrase = re.sub("\d+", " ", phrase)

    return phrase

def remove_stop(tokens) :
    """
    remove stop words from a column of texts
    """
    
    not_stop_words = [word for word in tokens if word not in sw]
    return not_stop_words
 
def remove_punctuation(text) : 
    """
    remove punctuation from a column of texts
    """
    return("".join([ch for ch in text if ch not in punctuation]))

def tokenize(text) : 
    """ Splitting on whitespace"""
    
    # modify this function to return tokens
    tokens = re.split(whitespace_pattern, text)
    return(tokens)


def remove_whitespace_token(tokens):
    """ Remove whitespace tokens"""
    
    # loop through each token to find whitespace token and remove
    for i in tokens:
        if '' in tokens:
            tokens.remove('')
    return tokens

def prepare(text, pipeline) :
    """
    prepare function applies each cleaning transformation
    function onto a column of text
    """
    tokens = str(text)
    
    for transform in pipeline : 
        tokens = transform(tokens)
        
    return(tokens)


# list of cleaning functions
pipeline = [str.lower, decontracted, remove_punctuation, tokenize, remove_whitespace_token, remove_stop]



## Load tfidf model for new data later on
tfidf = joblib.load("tfidf.pkl")

# Load final model
final_model = joblib.load("final_svm_model.pkl")


# stream lit
st.title("Sentiment Analysis for Streaming Devices on Amazon")
with st.form("key1"):
    # ask for input
    review_input = st.text_area('Enter review here: ')
    button_check = st.form_submit_button("Submit")


if review_input != '':

    # Pre-process amazon review
    review =  pd.DataFrame([review_input], columns=['review_text'])
    
    # Start with one review:
    text = review.review_text[0]
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords=sw, colormap = 'RdYlGn').generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

    review['clean_reviews'] = review['review_text'].apply(prepare,pipeline= pipeline)
    
    # remove any unicode characters
    review['clean_reviews'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

    # drop original reviews column
    review.drop(columns = ['review_text'], axis = 1, inplace = True)

    # untokenize plot descriptions
    review['clean_reviews'] = review['clean_reviews'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))

    review_input = tfidf.transform(review['clean_reviews'])

    # Make predictions
    with st.spinner('Predicting...'):
        review_pred = final_model.predict(review_input)[0]
        
    # Show predictions
    st.subheader('Prediction: '+ review_pred)
    pred_prob = np.round(final_model.predict_proba(review_input),3).ravel().tolist()
    pred_prob_df = pd.DataFrame({'Likelihood':pred_prob, 'Sentiment': final_model.classes_})

   # use plotly to create a bar graph of number of reviews by brand and using customized color coding 
    fig = px.bar(pred_prob_df, x="Sentiment", y="Likelihood", color="Sentiment", 
             title= "Likelihood of Predicted Sentiment",
             color_discrete_map={'Negative':'red',  'Positive':'rgb(27,158,119)'})
    fig.update_layout(autosize=False, width=800, height=600)
    st.plotly_chart(fig)