import re
import emoji
import num2words

from bs4 import BeautifulSoup

from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


def preprocess_tweet(tweet,
                     language="english",
                     remove_stopwords=True,
                     remove_hashtags=True,
                     handle_abbreviations=True,
                     lemmatize=True):
    """
    A function that takes a tweet as a string and some additional parameters and
    returns the tweet processed accordingly to the parameters provided
    """
    tweet = emoji.demojize(tweet, use_aliases=False, delimiters=(":", ":"))
    tweet = tweet.replace(":", " ")
    tweet = tweet.replace('"', '')
    # removing hashtags
    if remove_hashtags:
        tweet = re.sub(r'#\w+ ?', '', tweet)

    # user and URL handling
    tweet = re.sub(r'@\w+ ?', '', tweet)
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)

    # lowercase
    tweet = tweet.lower()

    # remove punctuations and other NON-ASCII characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'(\d+)', lambda w: num2words.num2words(int(w.group(0))), tweet)

    # HTML tags handling
    tweet = BeautifulSoup(tweet, "lxml").text

    # tokenizing the sentence for further pre-processing on word level
    tokenizer = TweetTokenizer()
    words = tokenizer.tokenize(tweet)

    # abbreviations handling
    if handle_abbreviations:
        ABBREVIATIONS = {"aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not",
                         "doesn't": "does not", "don't": "do not",
                         "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                         "he'll": "he will", "he's": "he is",
                         "i'd": "I would", "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is",
                         "it'll": "it will",
                         "i've": "I have", "let's": "let us", "mightn't": "might not", "mustn't": "must not",
                         "shan't": "shall not", "she'd": "she would",
                         "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that's": "that is",
                         "there's": "there is", "they'd": "they would",
                         "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would",
                         "we're": "we are", "weren't": "were not",
                         "we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is",
                         "what've": "what have", "where's": "where is",
                         "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is",
                         "who've": "who have", "won't": "will not",
                         "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are",
                         "you've": "you have", "'re": " are",
                         "wasn't": "was not", "we'll": " will"}
        words = [ABBREVIATIONS if word in ABBREVIATIONS else word for word in words]

    tweet = ' '.join(words)
    words = tokenizer.tokenize(tweet)

    # lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ,
                       "R": wordnet.ADV}
        pos_tagged_words = pos_tag(words)
        words = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_words]

    # stopwords removing
    if remove_stopwords:
        stopwords_ = set(stopwords.words(language))
        words = [word for word in words if word not in stopwords_]

    words = [word for word in words if len(word) > 1]

    forbidden_words = ['rt', 'nan', 'null']
    words = [word for word in words if word not in forbidden_words]

    final_tweet = " ".join(words)
    final_tweet = re.sub("  ", " ", final_tweet)

    return final_tweet
