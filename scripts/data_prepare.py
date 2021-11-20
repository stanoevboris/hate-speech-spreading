import pandas
import pandas as pd
import csv
from scripts.data_preprocessing import preprocess_tweet
from nltk.tokenize import TweetTokenizer

TWEETS_HS_DATA = \
    'data/profiling-hate-speech-spreaders-twitter/pan21-author-profiling-training-2021-03-14/en/tweets-hs-spreaders.csv'
TWEETS_CSV = 'data/tweets.csv'
USERS_CSV = 'data/users.csv'
WORDS_CSV = 'data/words.csv'
USERS_DICT = 'data/profiling-hate-speech-spreaders-twitter/pan21-author-profiling-training-2021-03-14/en/truth.txt'


def load_user_dict(filepath: str):
    # find each user given
    users_dict = {}
    with open(USERS_DICT, encoding='utf-8') as f:
        for line in f.readlines():
            line_data = line.replace('\n', '').split(':::')
            users_dict[line_data[0]] = int(line_data[1])
    return users_dict


def load_user_mappings():
    mapped_users = {}
    users_df = pd.read_csv(USERS_CSV)

    for index, row in users_df[['ID', 'USER_ID']].iterrows():
        id = row[0]
        user_id = row[1]
        mapped_users[user_id] = id

    return mapped_users


def create_user_mappings(filepath: str, users_ids: list, users_dict: dict):
    # create csv file with users information about their real ids and labels
    with open(filepath, 'w') as users_csv:
        writer = csv.writer(users_csv, delimiter=",")
        writer.writerow(('ID', 'USER_ID', 'LABEL'))
        for index, user in enumerate(users_ids):
            label = users_dict[user]
            writer.writerow((index, user, label))


def create_tweets_csv(dst_filepath: str):
    df = pd.read_csv(TWEETS_HS_DATA)
    mapped_users = load_user_mappings()
    with open(dst_filepath, 'w', encoding='utf-8') as twitters_csv:
        writer = csv.writer(twitters_csv, delimiter=",")
        writer.writerow(('ID', 'USER_ID', 'RAW_TWEET', 'PREPROCESSED'))
        tweet_id = 0
        for index, row in df.iterrows():
            user_id = row[0]
            raw_text = row[1]
            user_map_id = mapped_users[user_id]
            text = preprocess_tweet(raw_text)
            if text and len(text) > 0:
                writer.writerow((tweet_id, user_map_id, raw_text, text))
                tweet_id = tweet_id + 1


def extract_all_words(tweets_df: pd.DataFrame):
    tokenizer = TweetTokenizer()
    words = set()
    for index, row in tweets_df.iterrows():
        text = row[3]
        tweet_words = tokenizer.tokenize(text)
        for word in tweet_words:
            words.add(word)
    return words


def create_word_mappings(dst_filepath: str):
    tweets_df = pd.read_csv(TWEETS_CSV)
    words = extract_all_words(tweets_df)

    with open(dst_filepath, 'w', encoding='utf-8') as words_csv:
        writer = csv.writer(words_csv, delimiter=",")
        writer.writerow(('ID', 'WORD'))
        word_id = 0
        for word in words:
            writer.writerow((word_id, word))
            word_id = word_id + 1


def generate_users_tweets_edges(dst_filepath: str):
    tweets_df = pd.read_csv(TWEETS_CSV)
    users = tweets_df['USER_ID'].tolist()
    tweets = tweets_df['ID'].tolist()

    with open(dst_filepath, 'w', encoding='utf-8') as edges_csv:
        writer = csv.writer(edges_csv, delimiter=',')
        writer.writerow(('USER_ID', 'TWEET_ID'))
        for user, tweet in zip(users, tweets):
            writer.writerow((user, tweet))


def generate_tweets_words_edges(dst_filepath: str):
    tokenizer = TweetTokenizer()

    tweets_df = pd.read_csv(TWEETS_CSV, usecols=['ID', 'PREPROCESSED'], header=0)
    words_df = pd.read_csv(WORDS_CSV)
    words_dict = dict(words_df[['WORD', 'ID']].values)
    with open(dst_filepath, 'w', encoding='utf-8') as edges_csv:
        writer = csv.writer(edges_csv, delimiter=',')
        writer.writerow(('TWEET_ID', 'WORD_ID'))

        for index, row in tweets_df.iterrows():
            tweet_id = row[0]
            tweet = row[1]
            tweet_words = tokenizer.tokenize(tweet)

            for word in tweet_words:
                word_id = words_dict[word]
                writer.writerow((tweet_id, word_id))


def aggregate_tweets_on_user_level() -> pandas.DataFrame:
    tweets_df = pd.read_csv(TWEETS_CSV)
    users_df = pd.read_csv(USERS_CSV)

    tweets_df_agg = tweets_df[['USER_ID', 'RAW_TWEET']].groupby('USER_ID')['RAW_TWEET'].agg(
        lambda x: list(x.astype(str))).reset_index()
    df = tweets_df_agg.join(users_df, on='USER_ID', how='inner', lsuffix='_left', rsuffix='_right')
    df = df[['USER_ID', 'RAW_TWEET', 'LABEL']]

    return df
