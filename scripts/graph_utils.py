import pandas as pd
import torch as th
import dgl
from scripts.embeddings import load_embeddings

USER_TWEETS_EDGES = 'data/users-written-tweets.csv'
TWEETS_WORDS_EDGES = 'data/tweet-contains-words.csv'
USERS_EMBEDDINGS = 'data/users_embeddings.pkl'
TWEETS_EMBEDDINGS = 'data/tweets_embeddings.pkl'
WORDS_EMBEDDINGS = 'data/words_embeddings.pkl'
USERS_CSV = 'data/users.csv'

def load_user_tweet_edges():
    user_tweets_df = pd.read_csv(USER_TWEETS_EDGES, delimiter=',', header=0)

    users = th.tensor(user_tweets_df['USER_ID'].tolist())
    tweets = th.tensor(user_tweets_df['TWEET_ID'].tolist())
    user_tweets_edges = (users, tweets)
    tweets_user_edges = (tweets, users)

    return user_tweets_edges, tweets_user_edges


def load_tweet_words_edges():
    tweets_words_df = pd.read_csv(TWEETS_WORDS_EDGES, delimiter=',', header=0)

    tweets = th.tensor(tweets_words_df['TWEET_ID'].tolist())
    words = th.tensor(tweets_words_df['WORD_ID'].tolist())
    tweets_words_edges = (tweets, words)
    words_tweets_edges = (words, tweets)

    return tweets_words_edges, words_tweets_edges


def create_heterograph():
    user_tweets_edges, tweets_user_edges = load_user_tweet_edges()
    tweets_words_edges, words_tweets_edges = load_tweet_words_edges()

    graph_data = {
        ('user', 'writes', 'tweet'): user_tweets_edges,
        ('tweet', 'written-by', 'user'): tweets_user_edges,
        ('tweet', 'contains', 'word'): tweets_words_edges,
        ('word', 'belongs-to', 'tweet'): words_tweets_edges
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

    dgl.save_graphs('graphs/heterograph.bin', graph)
