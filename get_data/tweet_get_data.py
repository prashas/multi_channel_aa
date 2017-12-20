from collections import Counter
import itertools
from operator import itemgetter

from keras.utils import np_utils
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer #, base_filter
from keras.preprocessing import sequence
import numpy
# fix random seed for reproducibility
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from util_files.utils import ark_tweet_tokenizer, ctr_remove_below_threshold


seed = 7
numpy.random.seed(seed)


__author__ = 'shrprasha'

import util_files.file_utils as futils
import os
from amazon_get_data import get_all_authors_train_test as amazon_get_all_authors_train_test, \
    get_all_authors_create_train_test


def convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size=0.2):
    le = preprocessing.LabelEncoder()
    if val_size:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=val_size, random_state=190)
        le.fit(Y_train)
        Y_train = list(le.transform(Y_train))
        Y_val = list(le.transform(Y_val))
        Y_test = list(le.transform(Y_test))
        return X_train, np_utils.to_categorical(Y_train), X_val, np_utils.to_categorical(Y_val), X_test, np_utils.to_categorical(Y_test)
    else:
        le.fit(Y_train)
        Y_train = list(le.transform(Y_train))
        Y_test = list(le.transform(Y_test))
        # Y_test not converted to one-hot since there were problems with getting accuracy by comparing one-hot with probs
        return X_train, np_utils.to_categorical(Y_train), X_test, np_utils.to_categorical(Y_test)



# TESTED AFTER MAKING THREE FXNS; 08/14 4p
def get_authors_train_test_koppel_vauthors(rootdir, authors, test_fold=None, val_size=0):
    X_train, X_test, Y_train, Y_test = get_all_authors_train_test_fold(rootdir, authors, test_fold, total_docs=200)
    return convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size)

def get_authors_train_test_koppel_combined_tweets(rootdir, authors, test_fold=None, val_size=0, combined=4, total_docs=200):
    X_train, X_test, Y_train, Y_test = get_all_authors_train_test_fold(rootdir, authors, test_fold, combined=combined, total_docs=total_docs)
    return convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size)

# NOT TESTED AFTER MAKING THREE FXNS; 08/14 4p
def get_authors_train_test_koppel_vtweets(rootdir, authors, size=None, val_size=0):
    X_train, X_test, Y_train, Y_test = get_all_authors_train_test(rootdir, authors, size)
    return convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size)

def get_authors_train_test_amazon_from_partitions(rootdir, authors, val_size=0, sents_count=5, chars_count=0):
    X_train, X_test, Y_train, Y_test = amazon_get_all_authors_train_test(rootdir, authors, sents_count, chars_count)
    return convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size)

def get_authors_train_test_amazon(rootdir, authors, val_size=0.2):
    X_train, X_test, Y_train, Y_test = get_all_authors_create_train_test(rootdir, authors)
    return convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size)


def get_char_ngrams_fxn(ngram):
    def char_ngrams(text):
        return [text[i:i+ngram] for i in range(len(text)-ngram+1)]
    return char_ngrams


# created in order to store word_to_idx, for eg. when we have validation and test sets
# also, makes more sense
# truncates any test data > max_len of train
class Features:

    def __init__(self, feature_type, min_df=0, max_feats=None, max_len=None):
        # print feature_type
        if feature_type == "word":
            self.feature_fxn = ark_tweet_tokenizer
        # elif feature_type == "char_1":
        #     self.feature_fxn = get_char_ngrams_fxn(ngram=1)
        # elif feature_type == "char_2":
        #     self.feature_fxn = get_char_ngrams_fxn(ngram=2)
        # elif feature_type == "char_3":
        #     self.feature_fxn = get_char_ngrams_fxn(ngram=3)
        elif feature_type.startswith("char"):
            print feature_type
            self.feature_fxn = get_char_ngrams_fxn(ngram=int(feature_type.split("_")[-1]))
        self.min_df = min_df
        self.max_feats = max_feats
        self.max_len = max_len

    @property
    def feature_map(self):
        return self.feature_map

    @feature_map.setter
    def feature_map(self, feature_map):
        self.feature_map = feature_map

    def fit(self, X):
        X_tokenized = [self.feature_fxn(t) for t in X]
        all_words_ctr = Counter(itertools.chain.from_iterable(X_tokenized))
        if self.min_df:
            print "before filter by threshold:", len(all_words_ctr)
            ctr_remove_below_threshold(all_words_ctr, self.min_df)
            print "after filter by threshold:", len(all_words_ctr)
        if self.max_feats:
            all_words_ctr = Counter({k:v for k, v in all_words_ctr.most_common(self.max_feats)})
        all_words = ["<PAD>"] + all_words_ctr.keys()  # reserving 0 for padding
        self.feature_map = dict(zip(all_words, range(len(all_words))))
        if not self.max_len:
            self.max_len = max(len(s) for s in X_tokenized)

    def transform(self, X):
        X_tokenized = [self.feature_fxn(t) for t in X]
        X = []
        for tweet in X_tokenized:
            X.append([self.feature_map[w] for w in tweet if w in self.feature_map])
        X = sequence.pad_sequences(X, maxlen=self.max_len)
        return X

    # do not need to tokenize twice
    def fit_transform(self, X):
        X_tokenized = [self.feature_fxn(t) for t in X]
        # all_words = list(set(itertools.chain.from_iterable(X_tokenized)))
        all_words_ctr = Counter(itertools.chain.from_iterable(X_tokenized))
        print "before filter by threshold:", len(all_words_ctr)
        ctr_remove_below_threshold(all_words_ctr, self.min_df)
        print "after filter by threshold:", len(all_words_ctr)
        all_words = ["<PAD>"] + all_words_ctr.keys()  # reserving 0 for padding
        self.feature_map = dict(zip(all_words, range(len(all_words))))
        self.max_len = max(len(s) for s in X_tokenized)
        X = []
        for tweet in X_tokenized:
            X.append([self.feature_map[w] for w in tweet if w in self.feature_map])
        X = sequence.pad_sequences(X, maxlen=self.max_len)
        return X


####from cross_AA###


# for varying tweets, need to fix getting folds for varying tweets as well
def get_all_authors_train_test(rootdir, authors, size):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for author in authors:
        train_tweets, test_tweets = get_author_text_train_test(rootdir, author, size)
        X_train.extend(train_tweets)
        X_test.extend(test_tweets)
        Y_train.extend([author] * len(train_tweets))
        Y_test.extend([author] * len(test_tweets))
    return X_train, X_test, Y_train, Y_test


# for varying authors and for combined
def get_all_authors_train_test_fold(rootdir, authors, test_fold, combined=1, total_docs=None):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    docs_per_fold = total_docs / 10 if total_docs else None
    for author in authors:
        train_tweets, test_tweets = get_author_text_train_test_fold(rootdir, author, test_fold, combined, docs_per_fold)
        # print author
        # print train_tweets
        # print test_tweets
        X_train.extend(train_tweets)
        X_test.extend(test_tweets)
        Y_train.extend([author] * len(train_tweets))
        Y_test.extend([author] * len(test_tweets))
    return X_train, X_test, Y_train, Y_test



def get_author_text_train_test(rootdir, author, size):
    single_size = size / 10
    author_rootdir = os.path.join(rootdir, author)
    train_tweets = []
    for fname in futils.get_files_in_folder(author_rootdir):
        tweets = list(futils.get_lines_in_file(os.path.join(author_rootdir, fname)))
        if fname.endswith("_0.txt"):
            test_tweets = [tweet.strip() for tweet in tweets[:single_size]]
        else:
            train_tweets.extend([tweet.strip() for tweet in tweets[:single_size]])
    return train_tweets, test_tweets


def combine_docs(docs, combined):
    return [" ".join(docs[i*combined:(i+1)*combined]) for i in range(len(docs) / combined)]


def get_author_text_train_test_fold(rootdir, author, test_fold=0, combined=1, docs_per_fold=None):
    author_rootdir = os.path.join(rootdir, author)
    train_tweets = []
    for fname in futils.get_files_in_folder(author_rootdir):
        tweets = [tweet.strip() for tweet in futils.get_lines_in_file(os.path.join(author_rootdir, fname))]
        if combined > 1:
            tweets = combine_docs(tweets, combined)
        if fname.endswith("_%d.txt" % test_fold):
            test_tweets = tweets[:docs_per_fold]
        else:
            train_tweets.extend(tweets[:docs_per_fold])
    return train_tweets, test_tweets


####from cross_AA###

## does not work with unicode
## X=X_train+X_test
# in caller:
#     converted, max_len = convert_dataset_to_words(X_train + X_test)
#     X_train = converted[:len(X_train),:]
#     X_test = converted[-len(X_test):,:]
def convert_dataset_to_words_keras(X):
    # print X[0], len(X)
    print X

    ## does not work with unicode
    tk = Tokenizer(nb_words=2000, filters=[], lower=True, split=" ")
    tk.fit_on_texts(X[:2])
    X = tk.texts_to_sequences(X)

    max_len = max(len(s) for s in X)
    print max_len
    X = sequence.pad_sequences(X, maxlen=max_len)
    return X, max_len


def convert_dataset_to_words_stfwd(X_train, X_test):
    X_train_tokenized = [ark_tweet_tokenizer(t) for t in X_train]
    X_test_tokenized = [ark_tweet_tokenizer(t) for t in X_test]
    all_words = list(set(itertools.chain.from_iterable(X_train_tokenized)))
    word_to_idx = dict(zip(all_words, range(len(all_words))))

    X_train = []
    X_test = []
    for tweet in X_train_tokenized:
        X_train.append([word_to_idx[w] for w in tweet])
    for tweet in X_test_tokenized:
        X_test.append([word_to_idx[w] for w in tweet if w in word_to_idx])

    max_len = max(len(s) for s in X_train+X_test)
    print "max_len", max_len
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    return X_train, X_test, word_to_idx

# will only work for perceptron; i.e. no embedding
def convert_dataset_to_words(X_train, X_test):
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), tokenizer=ark_tweet_tokenizer)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)
    return X_train.todense(), X_test.todense()


def get_word_features(rootdir, authors_50, size):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for author in authors_50:
        train_tweets, test_tweets = get_author_text_train_test(rootdir, author, size)
        X_train.extend(train_tweets)
        X_test.extend(test_tweets)
        Y_train.extend([author] * len(train_tweets))
        Y_test.extend([author] * len(test_tweets))
    X_train, X_test, word_to_idx = convert_dataset_to_words_stfwd(X_train, X_test)
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y_train = list(le.transform(Y_train))
    Y_test = list(le.transform(Y_test))
    # print Y_test
    # Y_test not converted to one-hot since there were problems with getting accuracy by comparing one-hot with probs
    return X_train, np_utils.to_categorical(Y_train), X_test, np_utils.to_categorical(Y_test), word_to_idx

######from get_data


def get_ngrams_from_text(text, ngram=2):
    return [text[i:i+ngram] for i in range(len(text)-ngram+1)]

def convert_dataset_to_ngrams_stfwd(X_train, X_test):
    X_train_ngrams = [get_ngrams_from_text(x) for x in X_train]
    X_test_ngrams = [get_ngrams_from_text(x) for x in X_test]
    all_ngrams = list(set(itertools.chain.from_iterable(X_train_ngrams)))
    ngram_to_idx = dict(zip(all_ngrams, range(len(all_ngrams))))
    X_train = []
    X_test = []
    for tweet in X_train_ngrams:
        X_train.append([ngram_to_idx[w] for w in tweet])
    for tweet in X_test_ngrams:
        X_test.append([ngram_to_idx[w] for w in tweet if w in ngram_to_idx])

    max_len = max(len(s) for s in X_train+X_test)
    print "max_len", max_len
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    return X_train, X_test, ngram_to_idx


def get_char_bigram_features(rootdir, authors, size):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for author in authors:
        train_tweets, test_tweets = get_author_text_train_test(rootdir, author, size)
        X_train.extend(train_tweets)
        X_test.extend(test_tweets)
        Y_train.extend([author] * len(train_tweets))
        Y_test.extend([author] * len(test_tweets))
    X_train, X_test, word_to_idx = convert_dataset_to_ngrams_stfwd(X_train, X_test)
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y_train = list(le.transform(Y_train))
    Y_test = list(le.transform(Y_test))

    # Y_test not converted to one-hot since there were problems with getting accuracy by comparing one-hot with probs
    return X_train, np_utils.to_categorical(Y_train), X_test, np_utils.to_categorical(Y_test), word_to_idx
    # return X_train, np_utils.to_categorical(Y_uid_train), numpy.array(Y_gender_train), X_test, \
    #        np_utils.to_categorical(Y_uid_test), numpy.array(Y_gender_test), word_to_idx


def get_char_bigram_features_author(rootdir, authors, test_fold):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for author in authors:
        train_tweets, test_tweets = get_author_text_train_test_fold(rootdir, author, test_fold)
        X_train.extend(train_tweets)
        X_test.extend(test_tweets)
        Y_train.extend([author] * len(train_tweets))
        Y_test.extend([author] * len(test_tweets))
    X_train, X_test, word_to_idx = convert_dataset_to_ngrams_stfwd(X_train, X_test)
    le = preprocessing.LabelEncoder()
    le.fit(Y_train)
    Y_train = list(le.transform(Y_train))
    Y_test = list(le.transform(Y_test))
    return X_train, np_utils.to_categorical(Y_train), X_test, np_utils.to_categorical(Y_test), word_to_idx



if __name__ == "__main__":
    X_train = ["this is sparta", "this is not", "this is sparta", "this is not", "this is sparta", "this is not", "this is sparta", "this is not"]
    X_test = ["this is nice"]
    # print convert_dataset_to_words_stfwd(X_train, X_test)
    # fxn = get_char_ngrams_fxn(1)
    # print fxn("this")

    print combine_docs(X_train, 3)


# def get_author_text_train_test_fold(rootdir, author, test_fold=0, combined=1, docs_per_fold=None):


    # expt_type = "varying_training_set_size"
    #
    # setting = "varying_tweets.txt"
    #
    # rootdir = os.path.join(os.path.join(resources_rootdir, "koppel_tweets", "cleaned_all", expt_type))
    # authors_500 = [line.strip() for line in
    #                futils.get_lines_in_file(os.path.join(resources_rootdir, "koppel_tweets", setting))]
    # authors_50 = authors_500[:10]


