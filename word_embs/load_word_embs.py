import os

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from config import resources_rootdir, res_basedir
from get_data.tweet_get_data import Features, get_authors_train_test_koppel_vauthors
from gensim.models.wrappers import FastText


# created separately to call again for combined
def get_fasttext_model(dataset="tweet", model_type="bin"):
    w2v_rootdir = os.path.join(res_basedir, "word2vecs")
    tweets_rootdir = os.path.join(resources_rootdir, "tweet_w2v", "tweet_fasttext")
    ds_rootdir = os.path.join(resources_rootdir, "ds_aa", "fasttext_embs")
    amazon_rootdir = os.path.join(resources_rootdir, "amazon", "fasttext_embs")
    # amazon_rootdir = os.path.join(resources_rootdir, "amazon", "fasttext_embs_50_eps")
    if dataset == "tweet":
        model_path = os.path.join(tweets_rootdir, "tweet_fasttext.{}".format(model_type))
    elif dataset == "ds":
        model_path = os.path.join(ds_rootdir, "ds_fasttext.{}".format(model_type))
    elif dataset == "amazon":
        model_path = os.path.join(amazon_rootdir, "amazon_fasttext.{}".format(model_type))
    elif dataset == "wiki":
        model_path = os.path.join(w2v_rootdir, "wiki.en/wiki.en.{}".format(model_type))
    elif dataset == "simple":
        model_path = os.path.join(w2v_rootdir, "wiki.simple/wiki.simple.{}".format(model_type))
    print "fasttext model: ", model_path

    if model_type == "bin":
        model = FastText.load_fasttext_format(model_path)
    else:
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model


# TESTED?; THIS FUNCTION ON ALL COMBOS; 10/02 03:00A
def get_fasttext_model_emb(dataset="tweet", wiki_combined=True, model_type="bin"):
    model = get_fasttext_model(dataset, model_type)
    # wiki combined only works with bin
    if model_type == "vec" or dataset == "wiki":  # failsafe
        wiki_combined = False
    if wiki_combined:
        print "wiki combined model"
        wiki_model = get_fasttext_model(dataset="wiki", model_type=model_type)
        return get_word_embedding_fasttextbin(wiki_model, model)
    else:
        return get_word_embedding_fasttextbin(model) if model_type=="bin" else get_word_embedding_txtvector(model)


# TODO: change args everywhere to wiki_fasttext for wiki fasttext model: eg wiki_fasttext_bin
# TODO: change args everywhere to *_vec for vec model: eg. tweet_fasttext_vec
def get_w2vmodel(w2v_type=None, dumped_emb_fpath=None, wiki_combined=True):
    w2v_rootdir = os.path.join(res_basedir, "word2vecs")
    print "w2v_type:", w2v_type
    if dumped_emb_fpath:
        print "dumped_emb_fpath:", dumped_emb_fpath
        return get_word_embedding_txtvector(KeyedVectors.load_word2vec_format(dumped_emb_fpath, binary=False))
    elif "fasttext" in w2v_type:
        dataset, _, model_type = w2v_type.split("_")
        return get_fasttext_model_emb(dataset, wiki_combined, model_type)
    elif w2v_type == 'google_w2v':
        return get_word_embedding_txtvector(KeyedVectors.load_word2vec_format(os.path.join(w2v_rootdir, "GoogleNews-vectors-negative300.bin"), binary=True))
    elif w2v_type == 'numberbatch':
        return get_word_embedding_txtvector(KeyedVectors.load_word2vec_format(os.path.join(w2v_rootdir, "numberbatch-en.txt"), binary=False))
    elif w2v_type == 'tweet_w2v':
        return get_word_embedding_txtvector(Word2Vec.load(os.path.join(w2v_rootdir, 'tweet_w2v', 'w2v_cbow_300_iter100')))
    else:
        raise ValueError('Type of Embeddings value misspelled')

def get_word_embedding_fasttextbin(w2vmodel, tweet_w2vmodel=None):
    def get_word_embedding(word):
        try:
            return w2vmodel[word]
        except:
            try:
                return tweet_w2vmodel[word]
            except:
                return None
    return get_word_embedding, w2vmodel.vector_size

# def get_word_embedding_fasttextbin(w2vmodel, tweet_w2vmodel=None):
#     def get_word_embedding(word):
#         try:
#             print "first try"
#             emb = w2vmodel[word]
#         except:
#             try:
#                 print "second try"
#                 emb = tweet_w2vmodel[word]
#             except:
#                 emb = None
#         return emb
#     return get_word_embedding, w2vmodel.vector_size
#
def get_word_embedding_txtvector(w2vmodel):
    emb_vocab = w2vmodel.vocab
    weight_matrix = w2vmodel.syn0
    embedding_dim = weight_matrix.shape[1]
    def get_word_embedding(word):
        try:
            embedding_word = emb_vocab.get(word)
            return weight_matrix[embedding_word.index]
        except:
            return None
    return get_word_embedding, embedding_dim

def create_embedding(word_to_idx, w2v_type, wiki_combined=True):
    # print w2v_type, os.path.splitext(w2v_type)
    if os.path.splitext(w2v_type)[-1] == ".vec":   # dumped model
        # print "dumped"
        get_word_embedding, embedding_dim = get_w2vmodel(dumped_emb_fpath=w2v_type)
    else:
        # print "loaded"
        get_word_embedding, embedding_dim = get_w2vmodel(w2v_type=w2v_type, wiki_combined=wiki_combined)
    # words not found in embedding index will be all-zeros.
    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
    oov_words = []
    for word, idx in word_to_idx.iteritems():
        emb = get_word_embedding(word)
        if emb is not None:
            embedding_matrix[idx] = emb
        else:
            oov_words.append(word)
    # takes too long
    # not_found = embedding_matrix.shape[0] - np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    # oov_words = [k for k, v in word_to_idx.iteritems() if v in np.where(~embedding_matrix.any(axis=1))[0]]
    # print "OOV: {}; {}%".format(not_found, not_found * 100. / len(word_to_idx))
    print "OOV: {}; {}%".format(len(oov_words), len(oov_words) * 100. / len(word_to_idx))
    print oov_words[:100]
    return [embedding_matrix]

# get {word: embedding} instead of [vocab_embeddings] for dumping all embs in a corpus beforehand
# also returns all oovs
# eg in word_embs/fasttext_incremental.py
def create_embedding_for_dump(words, w2v_type):
    get_word_embedding, embedding_dim = get_w2vmodel(w2v_type=w2v_type)
    # words not found in embedding index will be all-zeros.
    embeddings = {}
    oov_words = []
    for word in words:
        emb = get_word_embedding(word)
        if emb is not None:
            embeddings[word] = emb
        else:
            oov_words.append(word)
    print "OOV: {}; {}%".format(len(oov_words), len(oov_words) * 100. / len(words))
    return embeddings, oov_words


def get_idx_to_word(word_to_idx):
    return {v:k for k, v in word_to_idx.iteritems()}


def load_model(w2v_type):
    if os.path.splitext(w2v_type)[-1] == ".vec":   # dumped model
        get_word_embedding, embedding_dim = get_w2vmodel(dumped_emb_fpath=w2v_type, wiki_combined=False)
    else:
        get_word_embedding, embedding_dim = get_w2vmodel(w2v_type=w2v_type, wiki_combined=False)
    return get_word_embedding, embedding_dim


def create_doc_emb_repr(doc_wordidx, word_to_idx, get_word_embedding, embedding_dim):
    idx_to_word = get_idx_to_word(word_to_idx)
    # commented since will be called for each doc
    # if os.path.splitext(w2v_type)[-1] == ".vec":   # dumped model
    #     get_word_embedding, embedding_dim = get_w2vmodel(dumped_emb_fpath=w2v_type, wiki_combined=False)
    # else:
    #     get_word_embedding, embedding_dim = get_w2vmodel(w2v_type=w2v_type, wiki_combined=False)
    # if no words in the doc is found
    embedding_matrix = np.zeros((len(doc_wordidx), embedding_dim))
    oov_words = []
    for idx, wordidx in enumerate(doc_wordidx):
        word = idx_to_word[wordidx]
        emb = get_word_embedding(word)
        if emb is not None:
            embedding_matrix[idx] = emb
        else:
            oov_words.append(word)
    # print "OOV: {}; {}%".format(len(oov_words), len(oov_words) * 100. / len(doc_wordidx))
    return np.mean(embedding_matrix, axis=0)



if __name__ == "__main__":
    from initial.all_models_new import get_author_names

    expt_type = "varying_number_of_authors"
    rootdir = os.path.join(os.path.join(resources_rootdir, "koppel_tweets", "cleaned_all", expt_type))
    setting = "varying_authors_list"
    author_lists_fpath = os.path.join(resources_rootdir, "koppel_tweets", setting)
    authors = get_author_names(author_lists_fpath, 1000)
    # authors = get_author_names(author_lists_fpath, 1000)[:2]
    # dump_rootdir = "/home/ritual/resources/cross-AA/koppel_twitter/w2vdumps"

    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_authors_train_test_koppel_vauthors(rootdir, authors, test_fold=0, val_size=0.1)
    word_feats = Features("word")
    word_feats.fit(X_train)
    X_train_word, X_val_word, X_test_word, word_to_idx = word_feats.transform(X_train), word_feats.transform(X_val), word_feats.transform(X_test), word_feats.feature_map

    create_embedding(word_to_idx, w2v_type='fasttextbin')
    # create_embedding(word_to_idx, w2v_type='tweet_w2v')
    # create_embedding(word_to_idx, w2v_type='numberbatch')
    # create_embedding(word_to_idx, w2v_type='google_w2v')
    # create_embedding(word_to_idx, w2v_type='fasttext')

    # raise SystemExit(0)

    # word2vec_file = os.path.join(res_basedir, "GoogleNews-vectors-negative300.bin")
    # # word2vec_file = "/Users/shrprasha/Projects/resources/GoogleNews-vectors-negative300.bin"
    # w2vmodel = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    # embedding_index = w2vmodel.vocab
    # weight_matrix = w2vmodel.syn0
    #
    # print embedding_index["apple"]   # Vocab(count:2986533, index:13467)
    # print weight_matrix[embedding_index["apple"].index]

    # ftt = get_w2vmodel(w2v_type="fasttextbin")
