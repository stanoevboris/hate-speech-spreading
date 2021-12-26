import pandas as pd
import torch as th
import dgl
from scripts.embeddings import load_embeddings

USER_TWEETS_EDGES_EN = 'data/users-written-tweets_en.csv'
TWEETS_WORDS_EDGES_EN = 'data/tweet-contains-words_en.csv'
WORD_WORD_EDGES_EN = 'data/word-word-edges_en.csv'
TWEET_TWEET_EDGES_EN = 'data/tweet-tweet-edges_en.csv'
USERS_EMBEDDINGS_EN = 'data/users_embeddings_en.pkl'
TWEETS_EMBEDDINGS_EN = 'data/tweets_embeddings_en.pkl'
WORDS_EMBEDDINGS_EN = 'data/words_embeddings_en.pkl'
USERS_CSV_EN = 'data/users_en.csv'

USER_TWEETS_EDGES_ES = 'data/users-written-tweets_es.csv'
TWEETS_WORDS_EDGES_ES = 'data/tweet-contains-words_es.csv'
WORD_WORD_EDGES_ES = 'data/word-word-edges_es.csv'
TWEET_TWEET_EDGES_ES = 'data/tweet-tweet-edges_es.csv'
USERS_EMBEDDINGS_ES = 'data/users_embeddings_es.pkl'
TWEETS_EMBEDDINGS_ES = 'data/tweets_embeddings_es.pkl'
WORDS_EMBEDDINGS_ES = 'data/words_embeddings_es.pkl'
USERS_CSV_ES = 'data/users_es.csv'

HET_GRAPH_PATH_EN = 'graphs/heterograph_en.bin'
HET_GRAPH_PATH_ES = 'graphs/heterograph_es.bin'


def load_user_tweet_edges(language='en'):
    if language == 'en':
        USER_TWEETS_EDGES = USER_TWEETS_EDGES_EN
    else:
        USER_TWEETS_EDGES = USER_TWEETS_EDGES_ES

    user_tweets_df = pd.read_csv(USER_TWEETS_EDGES, delimiter=',', header=0)

    users = th.tensor(user_tweets_df['USER_ID'].tolist())
    tweets = th.tensor(user_tweets_df['TWEET_ID'].tolist())
    user_tweets_edges = (users, tweets)
    tweets_user_edges = (tweets, users)

    return user_tweets_edges, tweets_user_edges


def load_tweet_words_edges(language='en'):
    if language == 'en':
        TWEETS_WORDS_EDGES = TWEETS_WORDS_EDGES_EN
    else:
        TWEETS_WORDS_EDGES = TWEETS_WORDS_EDGES_ES

    tweets_words_df = pd.read_csv(TWEETS_WORDS_EDGES, delimiter=',', header=0)

    tweets = th.tensor(tweets_words_df['TWEET_ID'].tolist())
    words = th.tensor(tweets_words_df['WORD_ID'].tolist())
    tweets_words_edges = (tweets, words)
    words_tweets_edges = (words, tweets)

    return tweets_words_edges, words_tweets_edges


def load_tweet_tweet_edges(language='en'):
    if language == 'en':
        TWEET_TWEET_EDGES = TWEET_TWEET_EDGES_EN
    else:
        TWEET_TWEET_EDGES = TWEET_TWEET_EDGES_ES

    tweet_tweet_df = pd.read_csv(TWEET_TWEET_EDGES, delimiter=',', header=0)

    tweet1 = th.tensor(tweet_tweet_df['TWEET_ONE_ID'].tolist())
    tweet2 = th.tensor(tweet_tweet_df['TWEET_TWO_ID'].tolist())
    tweet1_tweet2_edges = (tweet1, tweet2)
    tweet2_tweet1_edges = (tweet2, tweet1)

    return tweet1_tweet2_edges, tweet2_tweet1_edges


def load_word_word_edges(language='en'):
    if language == 'en':
        WORD_WORD_EDGES = WORD_WORD_EDGES_EN
    else:
        WORD_WORD_EDGES = WORD_WORD_EDGES_ES

    word_word_df = pd.read_csv(WORD_WORD_EDGES, delimiter=',', header=0)

    word1 = th.tensor(word_word_df['WORD_ONE_ID'].tolist())
    word2 = th.tensor(word_word_df['WORD_TWO_ID'].tolist())
    word1_word2_edges = (word1, word2)
    word2_word1_edges = (word2, word1)

    return word1_word2_edges, word2_word1_edges


def create_heterograph(language='en'):
    if language == 'en':
        USERS_EMBEDDINGS = USERS_EMBEDDINGS_EN
        TWEETS_EMBEDDINGS = TWEETS_EMBEDDINGS_EN
        WORDS_EMBEDDINGS = WORDS_EMBEDDINGS_EN
        USERS_CSV = USERS_CSV_EN
        graph_path = HET_GRAPH_PATH_EN
    else:
        USERS_EMBEDDINGS = USERS_EMBEDDINGS_ES
        TWEETS_EMBEDDINGS = TWEETS_EMBEDDINGS_ES
        WORDS_EMBEDDINGS = WORDS_EMBEDDINGS_ES
        USERS_CSV = USERS_CSV_ES
        graph_path = HET_GRAPH_PATH_ES

    user_tweets_edges, tweets_user_edges = load_user_tweet_edges(language)
    tweets_words_edges, words_tweets_edges = load_tweet_words_edges(language)
    tweet1_tweet2_edges, tweet2_tweet1_edges = load_tweet_tweet_edges(language)
    word1_word2_edges, word2_word1_edges = load_word_word_edges(language)

    graph_data = {
        ('user', 'writes', 'tweet'): user_tweets_edges,
        ('tweet', 'written-by', 'user'): tweets_user_edges,
        ('tweet', 'contains', 'word'): tweets_words_edges,
        ('word', 'belongs-to', 'tweet'): words_tweets_edges,
        ('tweet', 'similar-to', 'tweet'): tweet1_tweet2_edges,
        ('tweet', 'similar-with', 'tweet'): tweet2_tweet1_edges,
        ('word', 'similar_to', 'word'): word1_word2_edges,
        ('word', 'similar_with', 'word'): word2_word1_edges
    }

    graph = dgl.heterograph(graph_data)

    users, user_sentences, user_embeddings = load_embeddings(USERS_EMBEDDINGS)
    tweets, tweet_sentences, tweet_embeddings = load_embeddings(TWEETS_EMBEDDINGS)
    words, words_sentences, word_embeddings = load_embeddings(WORDS_EMBEDDINGS)

    graph.nodes['user'].data['h'] = th.stack(user_embeddings)
    graph.nodes['tweet'].data['h'] = th.stack(tweet_embeddings)
    graph.nodes['word'].data['h'] = th.stack(word_embeddings)

    users_df = pd.read_csv(USERS_CSV, header=0)
    labels = users_df['LABEL'].tolist()
    graph.nodes['user'].data['labels'] = th.tensor(labels)

    dgl.save_graphs(graph_path, graph)
