from ds_get_data import get_author_text_train_test

__author__ = 'shrprasha'

import util_files.file_utils as futils
import util_files.utils as nlp_utils
import os
import textwrap

def get_first_x_sentences(review, x):
    return " ".join(nlp_utils.get_sentences(review)[:x])


def get_all_authors_train_test(rootdir, authors, sents_count=5, chars_count=None, break_long_words=False):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    tot_docs = 0
    for author in authors:
        train_reviews = futils.get_lines_in_file_small(os.path.join(rootdir, "train", author))
        test_reviews = futils.get_lines_in_file_small(os.path.join(rootdir, "test", author))

        if sents_count:
            train_reviews = [get_first_x_sentences(review, sents_count) for review in train_reviews]
            test_reviews = [get_first_x_sentences(review, sents_count) for review in test_reviews]
        elif chars_count:
            train_reviews = [textwrap.wrap(review, width=chars_count, break_long_words=break_long_words)[0] for review in train_reviews]
            test_reviews = [textwrap.wrap(review, width=chars_count, break_long_words=break_long_words)[0] for review in test_reviews]

        tot_docs += len(train_reviews) + len(test_reviews)
        X_train.extend(train_reviews)
        X_test.extend(test_reviews)
        Y_train.extend([author] * len(train_reviews))
        Y_test.extend([author] * len(test_reviews))
    return X_train, X_test, Y_train, Y_test


def get_all_authors_create_train_test(rootdir, authors):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    # print  "here", len(authors)
    for author in authors:
        train_reviews, test_reviews = get_author_text_train_test(rootdir, author, train_ratio=0.8)
        # print author, len(train_reviews)
        X_train.extend(train_reviews)
        X_test.extend(test_reviews)
        Y_train.extend([author] * len(train_reviews))
        Y_test.extend([author] * len(test_reviews))
    return X_train, X_test, Y_train, Y_test