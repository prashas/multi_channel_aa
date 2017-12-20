import textwrap

__author__ = 'shrprasha'
import util_files.file_utils as futils
import os
import itertools
from keras.utils import np_utils
from sklearn import preprocessing
from keras.preprocessing import sequence
import numpy
# fix random seed for reproducibility


seed = 7
numpy.random.seed(seed)

def get_authors(rootdir, n=0):
    all_authors = futils.get_files_in_folder(rootdir)
    return all_authors[:n] if n else all_authors

def get_author_text_train_test(rootdir, author_fname, train_ratio = 0.8):
    posts = list(futils.get_lines_in_file_small(os.path.join(rootdir, author_fname)))#[:10]
    # print author_fname, len(posts)
    train_posts_count = int(len(posts) * train_ratio)
    return posts[:train_posts_count], posts[train_posts_count:]

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


def get_data_for_simple_baseline_authorship_only(rootdir, authors):
    X_train = []
    X_test = []
    Y_uid_train = []
    Y_uid_test = []
    for author_fname in authors:
        train_posts, test_posts = get_author_text_train_test(rootdir, author_fname)
        uid, _ = author_fname.split("_")
        X_train.extend(train_posts)
        X_test.extend(test_posts)
        Y_uid_train.extend([uid] * len(train_posts))
        Y_uid_test.extend([uid] * len(test_posts))
    uid_le = preprocessing.LabelEncoder()
    uid_le.fit(Y_uid_train)
    Y_uid_train = list(uid_le.transform(Y_uid_train))
    Y_uid_test = list(uid_le.transform(Y_uid_test))

    return X_train, Y_uid_train, X_test, Y_uid_test


# gender prediction done for all posts of a user as a whole
# authorship prediction done per post
# so, two different types of train sets necessary if we go by old projects
# but not implemented for now: DONE
def get_data_for_simple_baseline_authorship_gender(rootdir, authors, chars_count=None, break_long_words=False):
    X_train = []
    X_test = []
    # X_train_gender = []
    # X_test_gender = []
    Y_uid_train = []
    Y_gender_train = []
    Y_uid_test = []
    Y_gender_test = []
    # for author_fname in authors[:8]:
    for author_fname in authors:
        train_posts, test_posts = get_author_text_train_test(rootdir, author_fname)
        if chars_count:
            train_posts = [textwrap.wrap(review, width=chars_count, break_long_words=break_long_words)[0] for review in train_posts]
            test_posts = [textwrap.wrap(review, width=chars_count, break_long_words=break_long_words)[0] for review in test_posts]
            # for review in train_posts:
            #     try:
            #         textwrap.wrap(review, width=chars_count, break_long_words=break_long_words)[0]
            #     except:
            #         print review

        uid, gender = author_fname.split("_", 1)
        X_train.extend(train_posts)
        X_test.extend(test_posts)
        Y_uid_train.extend([uid] * len(train_posts))
        Y_gender_train.extend([gender] * len(train_posts))
        Y_uid_test.extend([uid] * len(test_posts))
        Y_gender_test.extend([gender] * len(test_posts))
    # uid_le = preprocessing.LabelEncoder()
    # uid_le.fit(Y_uid_train)
    # Y_uid_train = list(uid_le.transform(Y_uid_train))
    # Y_uid_test = list(uid_le.transform(Y_uid_test))


    # 10/04 changed for gender_le
    # gender_le = preprocessing.LabelEncoder()
    # gender_le.fit(Y_gender_train)
    # print gender_le.classes_
    # Y_gender_train = list(gender_le.transform(Y_gender_train))
    # Y_gender_test = list(gender_le.transform(Y_gender_test))
    # return X_train, Y_uid_train, Y_gender_train, X_test, Y_uid_test, Y_gender_test, gender_le
    return X_train, Y_uid_train, Y_gender_train, X_test, Y_uid_test, Y_gender_test


def get_data_for_simple_baseline_authorship_gender_nole(rootdir, authors):
    X_train = []
    X_test = []
    # X_train_gender = []
    # X_test_gender = []
    Y_uid_train = []
    Y_gender_train = []
    Y_uid_test = []
    Y_gender_test = []
    for author_fname in authors:
        train_posts, test_posts = get_author_text_train_test(rootdir, author_fname)
        uid, gender = author_fname.split("_")
        X_train.extend(train_posts)
        X_test.extend(test_posts)
        Y_uid_train.extend([uid] * len(train_posts))
        Y_gender_train.extend([gender] * len(train_posts))
        Y_uid_test.extend([uid] * len(test_posts))
        Y_gender_test.extend([gender] * len(test_posts))
    return X_train, Y_uid_train, Y_gender_train, X_test, Y_uid_test, Y_gender_test


# same author's text in both train and test: need to change
def get_data_for_simple_baseline(rootdir, authors):
    X_train = []
    X_test = []
    # X_train_gender = []
    # X_test_gender = []
    Y_uid_train = []
    Y_gender_train = []
    Y_age_train = []
    Y_uid_test = []
    Y_gender_test = []
    Y_age_test = []
    for author_fname in authors:
        train_posts, test_posts = get_author_text_train_test(rootdir, author_fname)
        uid, gender, age = author_fname.split("_")
        X_train.extend(train_posts)
        X_test.extend(test_posts)
        Y_uid_train.extend([uid] * len(train_posts))
        Y_gender_train.extend([gender] * len(train_posts))
        Y_age_train.extend([age] * len(train_posts))
        Y_uid_test.extend([uid] * len(test_posts))
        Y_gender_test.extend([gender] * len(test_posts))
        Y_age_test.extend([age] * len(test_posts))
    uid_le = preprocessing.LabelEncoder()
    uid_le.fit(Y_uid_train)
    Y_uid_train = list(uid_le.transform(Y_uid_train))
    Y_uid_test = list(uid_le.transform(Y_uid_test))

    gender_le = preprocessing.LabelEncoder()
    gender_le.fit(Y_gender_train)
    print gender_le.classes_
    age_le = preprocessing.LabelEncoder()
    age_le.fit(Y_age_train)
    print age_le.classes_
    Y_gender_train = list(gender_le.transform(Y_gender_train))
    Y_gender_test = list(gender_le.transform(Y_gender_test))
    Y_age_train = list(age_le.transform(Y_age_train))
    Y_age_test = list(age_le.transform(Y_age_test))
    return X_train, Y_uid_train, Y_gender_train, Y_age_train, X_test, Y_uid_test, Y_gender_test, gender_le, Y_age_test, age_le

def get_char_bigram_features(rootdir, authors):
    X_train = []
    X_test = []
    Y_uid_train = []
    Y_gender_train = []
    Y_uid_test = []
    Y_gender_test = []
    for author_fname in authors:
        train_posts, test_posts = get_author_text_train_test(rootdir, author_fname)
        uid, gender = author_fname.split("_")
        X_train.extend(train_posts)
        X_test.extend(test_posts)
        Y_uid_train.extend([uid] * len(train_posts))
        Y_gender_train.extend([gender] * len(train_posts))
        Y_uid_test.extend([uid] * len(test_posts))
        Y_gender_test.extend([gender] * len(test_posts))
    X_train, X_test, word_to_idx = convert_dataset_to_ngrams_stfwd(X_train, X_test)
    uid_le = preprocessing.LabelEncoder()
    uid_le.fit(Y_uid_train)
    Y_uid_train = list(uid_le.transform(Y_uid_train))
    Y_uid_test = list(uid_le.transform(Y_uid_test))

    gender_le = preprocessing.LabelEncoder()
    gender_le.fit(Y_gender_train)
    Y_gender_train = list(gender_le.transform(Y_gender_train))
    Y_gender_test = list(gender_le.transform(Y_gender_test))

    # Y_test not converted to one-hot since there were problems with getting accuracy by comparing one-hot with probs
    return X_train, np_utils.to_categorical(Y_uid_train), numpy.array(Y_gender_train), X_test, \
           np_utils.to_categorical(Y_uid_test), numpy.array(Y_gender_test), word_to_idx
    # return X_train, np_utils.to_categorical(Y_uid_train), np_utils.to_categorical(Y_gender_train), X_test, \
    #        np_utils.to_categorical(Y_uid_test), np_utils.to_categorical(Y_gender_test), word_to_idx



if __name__ == "__main__":
    rootdir1 = "/Users/shrprasha/resources/cross_aa/ds_aa"
    authors = get_authors(rootdir1, 10)
    get_char_bigram_features(rootdir1, authors)