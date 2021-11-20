from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle

TWEETS_CSV = 'data/tweets.csv'
USERS_CSV = 'data/users.csv'
WORDS_CSV = 'data/words.csv'
TWEETS_WORDS_EDGES = 'data/tweet-contains-words.csv'
MODEL = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
USERS_EMBEDDINGS = 'data/users_embeddings.pkl'
TWEETS_EMBEDDINGS = 'data/tweets_embeddings.pkl'
WORDS_EMBEDDINGS = 'data/words_embeddings.pkl'


# MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those sub-word token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_word_vector(sent, idx, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token ids that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_model_and_tokenizer():
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)

    return tokenizer, model


def save_embeddings(filepath: str, ids: list, sentences: list, embeddings: list):
    with open(filepath, "wb") as fOut:
        pickle.dump({'ids': ids, 'sentences': sentences, 'embeddings': embeddings}, fOut,
                    protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(filepath):
    # Load sentences & embeddings from disc
    with open(filepath, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_users = stored_data['ids']
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']

    return stored_users, stored_sentences, stored_embeddings


def generate_embeddings_util(raw_sentences: list) -> list:
    embeddings = list()
    tokenizer, model = load_model_and_tokenizer()

    for index, sentences in enumerate(raw_sentences):
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        sentence_embeddings = torch.mean(sentence_embeddings, 0)
        embeddings.append(sentence_embeddings)
        print(f"Sentence {index} appended.", end='\r', flush=True)

    return embeddings


def generate_user_embeddings():
    users_df = pd.read_csv(USERS_CSV)
    tweets_df = pd.read_csv(TWEETS_CSV)

    tweets_df_agg = tweets_df[['USER_ID', 'RAW_TWEET', 'PREPROCESSED']] \
        .groupby('USER_ID')[['RAW_TWEET', 'PREPROCESSED']].agg(lambda x: list(x.astype(str))).reset_index()

    tweets_df_agg['USER_ID'] = pd.to_numeric(tweets_df_agg['USER_ID'])

    df = tweets_df_agg.join(users_df, on='USER_ID', how='inner', lsuffix='_left', rsuffix='_right')

    clean_sentences = df['PREPROCESSED'].tolist()
    raw_sentences = df['RAW_TWEET'].tolist()
    all_users = df['USER_ID'].tolist()
    embeddings = generate_embeddings_util(raw_sentences=raw_sentences)

    save_embeddings(USERS_EMBEDDINGS, all_users, clean_sentences, embeddings)


def generate_tweet_embeddings():
    tweets_df = pd.read_csv(TWEETS_CSV)

    all_tweets = tweets_df['ID'].tolist()
    raw_sentences = tweets_df['RAW_TWEET'].tolist()
    clean_sentences = tweets_df['PREPROCESSED'].tolist()
    embeddings = generate_embeddings_util(raw_sentences=raw_sentences)

    save_embeddings(TWEETS_EMBEDDINGS, all_tweets, clean_sentences, embeddings)


def generate_word_embeddings():
    tweet_words = pd.read_csv(TWEETS_WORDS_EDGES, header=0)
    tweets = pd.read_csv(TWEETS_CSV, header=0)
    words = pd.read_csv(WORDS_CSV, header=0)

    df = tweet_words.join(tweets, on='TWEET_ID', how='inner', lsuffix='_left', rsuffix='_right')
    df = df.join(words, on='WORD_ID', how='inner', lsuffix='_left', rsuffix='_right')
    words_agg = df[['WORD_ID', 'WORD', 'PREPROCESSED']].groupby(['WORD_ID', 'WORD'])['PREPROCESSED'].agg(
        lambda x: list(x.astype(str))).reset_index()

    layers = [-4, -3, -2, -1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, output_hidden_states=True)

    all_words = words_agg['WORD_ID'].tolist()
    all_sentences = words_agg['PREPROCESSED'].tolist()
    embeddings = list()

    for index, row in words_agg.iterrows():
        word = row['WORD']
        word_embeddings = list()
        for sentence in row['PREPROCESSED']:
            idx = get_word_idx(sentence, word)

            word_embedding = get_word_vector(sentence, idx, tokenizer, model, layers)
            word_embeddings.append(word_embedding)

        word_embeddings = torch.mean(torch.stack(word_embeddings), 0)

        embeddings.append(word_embeddings)
        print(f"Word {index} appended.", end='\r', flush=True)

    save_embeddings(WORDS_EMBEDDINGS, all_words, all_sentences, embeddings)


def tokenize(tokenizer, sentences, SEQ_LEN):
    tokens = tokenizer.encode_plus(sentences, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='pt')
    return tokens['input_ids'], tokens['attention_mask']
