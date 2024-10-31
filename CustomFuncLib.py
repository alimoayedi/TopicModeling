import pandas as pd
import numpy as np

import nltk

from tensorflow.keras.preprocessing import sequence

nltk.download('averaged_perceptron_tagger')


def load_data():
    trainDocs = pd.read_parquet('/content/trainDocsDictionary.parquet')
    trainTopics = pd.read_parquet('/content/trainTopics.parquet')
    valDocs = pd.read_parquet('/content/valDocsDictionary.parquet')
    valTopcis = pd.read_parquet('/content/valTopics.parquet')
    testDocs = pd.read_parquet('/content/testDocsDictionary.parquet')
    testTopics = pd.read_parquet('/content/testTopics.parquet')
    return trainDocs, trainTopics, valDocs, valTopcis, testDocs, testTopics

def trim_documents(doc, max_length):
    if len(doc) > max_length:
        return doc[:max_length]  # Keep first 256 terms
    else:
        return doc  # Keep as is

def join_tokens(tokens):
   return " ".join(tokens)

def white_space_splitter(text):
    return text.split()

def remove_high_correlated_tokens(cosine_sim_score, tokens_lst):
    to_remove = set()
    for term_1 in tokens_lst:
        for term_2 in tokens_lst:
            if term_1 != term_2 and cosine_sim_score[term_1, term_2] > 0.9:
                    to_remove.add(term_2)
    return [term for term in tokens_lst if term not in to_remove]

def vectorize(vocab_vec, tokens_lst):
    return [vocab_vec[term] for term in tokens_lst if term in vocab_vec]

def get_number_of_tokens(df_col):
    return len(set([item for sublist in df_col for item in sublist]))

# create a list of elements from columns of DF
def create_list(row):
    elements = row.iloc[:].tolist()
    return elements

def lemmatize(tokenized_text):
    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokenized_text)

    # Initialize WordNetLemmatizer
    lemmatizer = nltk.WordNetLemmatizer()

    # Lemmatize words using POS tags
    lemmas = []
    for word, pos_tag in pos_tags:
        # Map POS tags to WordNet tags
        if pos_tag.startswith('N'):
            wn_tag = 0  # Noun
        elif pos_tag.startswith('V'):
            wn_tag = 1  # Verb
        elif pos_tag.startswith('J'):
            wn_tag = 2  # Adjective
        elif pos_tag.startswith('R'):
            wn_tag = 3  # Adverb
        else:
            wn_tag = 4  # No specific tag

        # Lemmatize the word with WordNet
        lemmas.append(wn_tag)

    return lemmas

def padding(lst, maximum_length):
    return sequence.pad_sequences([lst], maxlen=maximum_length, padding='post')[0]

def concatenate_arrays(parent_lst):
    concatinated = np.array([])
    for item in range(len(parent_lst)):
        concatinated = np.append(concatinated, parent_lst[item])
    return concatinated

def make_tuple(lst, size):
    list_size = len(lst)
    if size == 2:
        return [(lst[index], lst[index+1]) for index in range(list_size - size + 1)]
    elif size == 3:
        return [(lst[index], lst[index+1], lst[index+2]) for index in range(list_size - size + 1)]
    else:
        raise ValueError("Error in the value of the size! Check the method!")

def get_unique_token_pairs(lst_docs):
    unique_pairs = []
    for doc in lst_docs:
        unique_pairs.extend(list(set(doc)))
    return list(set(unique_pairs))

def get_token_pairs_count(tuples_lst, doc_tuples_lst):
    return [doc_tuples_lst.count(tuple) for tuple in tuples_lst]


