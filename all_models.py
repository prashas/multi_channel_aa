from __future__ import division

import shutil
import sys
import time


import keras.preprocessing.sequence

seed = 7
import numpy as np
np.random.seed(seed)



import keras.backend as K
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Dense, Dropout, Embedding, Input, Reshape
from keras.layers.convolutional import Convolution1D, Conv1D
from keras.layers.core import Lambda
from keras.layers.noise import AlphaDropout
from keras.models import Model
from keras.optimizers import Adam

from get_data.ds_get_data import get_data_for_simple_baseline_authorship_gender
from get_data.tweet_get_data import Features, get_authors_train_test_koppel_vauthors, convert_data_for_nn_models, \
    get_authors_train_test_amazon
from word_embs.load_word_embs import create_embedding
np.set_printoptions(threshold=np.inf)

__author__ = 'shrprasha'


import os
from config import resources_rootdir, proj_basedir
import util_files.file_utils as futils
# from keras import initializations

from keras.initializers import RandomNormal

class AttLayer(Layer):
    def __init__(self, **kwargs):
        # self.init = initializations.get('normal')   #1.1.0
        self.init = RandomNormal('normal')    #2.0.3
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

# def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
#     desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
#     head_activations, head_words = head[:,:,:n], head[:,:,n:]
#     desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
#
#     # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
#     # activation for every head word and every desc word
#     activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
#     # make sure we dont use description words that are masked out
#     activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
#
#     # for every head word compute weights for every desc word
#     activation_energies = K.reshape(activation_energies,(-1,maxlend))
#     activation_weights = K.softmax(activation_energies)
#     activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))
#
#     # for every head word compute weighted average of desc words
#     desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
#     return K.concatenate((desc_avg_word, head_words))

class SimpleAttLayer(Layer):
    def __init__(self, **kwargs):
        # self.init = initializations.get('normal')   #1.1.0
        self.init = RandomNormal('normal')    #2.0.3
        #self.input_spec = [InputSpec(ndim=3)]
        super(SimpleAttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        context = .1*input_shape[3]
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])



def get_size_authors_from_processid_no_bots(processid, authors_500):
    # sizes = [1, 2, 3, 4, 5, 6]
    sizes = [2, 3, 4, 5, 6]
    author_groups_idxs = [0, 33, 69, 100, 136, 174, 208, 245, 276, 314, 355]
    # sizes = [10]
    size = sizes[processid % len(sizes)]  # total no. of train tweets per author
    author_group = processid // len(sizes)  # out of 10 author groups of 50 each, which one to take
    print "size: ", size, "author group: ", author_group
    authors = authors_500[author_groups_idxs[author_group]:author_groups_idxs[author_group + 1]]
    return size, authors


def max_1d(X):
    return K.max(X, axis=1)


def get_conv_layer(node_input, filter_tuple, activation_value):
    n_layers = len(filter_tuple)
    cnn_nodes = []
    for i in range(n_layers):
        cnn = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='valid', activation=activation_value, subsample_length=1)(node_input)
        cnn = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn)
        cnn_nodes.append(cnn)
    return cnn_nodes

def get_main_model(maxlen, embedding_dims, max_features, filter_tuple, nb_classes, dropout=0.25, activation_value='relu', weights=None):
    #activation_value must be relu or tanh
    main_input = Input(shape=(maxlen,), dtype='int32')
    x = Embedding(max_features, embedding_dims, weights=weights, input_length=maxlen)(main_input)
    if dropout > 0:
        x = Dropout(dropout)(x)
    list_cnn_nodes = get_conv_layer(x, filter_tuple, activation_value)
    if len(list_cnn_nodes)>1:
        list_cnn_nodes = layers.Concatenate()(list_cnn_nodes)
    else:
        list_cnn_nodes = list_cnn_nodes[0] #Fix this horrible trick
    if dropout > 0:
        list_cnn_nodes = AlphaDropout(dropout)(list_cnn_nodes)
    return list_cnn_nodes, main_input

def get_single_channel_model(no_of_labels, X_train, embedding_dims, char_to_idx, filter_tuple, dropout=0.25, activation_value='relu', weights=None):
    #activation_value must be relu or tanh
    main_input = Input(shape=(X_train.shape[1],), dtype='int32', name='main_input')
    x = Embedding(len(char_to_idx), embedding_dims, weights=weights, input_length=X_train.shape[1])(main_input)
    if dropout > 0:
        x = Dropout(dropout)(x)
    list_cnn_nodes = get_conv_layer(x, filter_tuple, activation_value)
    if len(list_cnn_nodes)>1:
        list_cnn_nodes = layers.Concatenate()(list_cnn_nodes)
    else:
        list_cnn_nodes = list_cnn_nodes[0]
    if dropout > 0:
        list_cnn_nodes = AlphaDropout(dropout)(list_cnn_nodes)
    main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(list_cnn_nodes)
    model = Model(input=main_input, output=main_loss)
    return model

def get_multichannel_model(no_of_labels, X_train_channel1, X_train_channel2, channel1_to_idx, channel2_to_idx, embedding_dims, filter_tuple, activation_value, dropout=0.25, channel1_weights=None, channel2_weights=None):
    channel1_model, channel1_input = get_main_model(X_train_channel1.shape[1], embedding_dims, len(channel1_to_idx), filter_tuple, no_of_labels, dropout=dropout, activation_value=activation_value, weights=channel1_weights)
    channel2_model, channel2_input = get_main_model(X_train_channel2.shape[1], embedding_dims, len(channel2_to_idx), filter_tuple, no_of_labels, dropout=dropout, activation_value=activation_value, weights=channel2_weights)
    # merged_layer = merge([channel1_model, channel2_model], mode='concat', concat_axis=1, name='merged')
    merged_layer = layers.Concatenate(axis=1, name='merged')([channel1_model, channel2_model])
    main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(merged_layer)
    model = Model(input=[channel1_input, channel2_input], output=main_loss)
    return model

# same params shared convnet
def get_multichannel_model_shared(no_of_labels, X_train_channel1, X_train_channel2, channel1_to_idx, channel2_to_idx, embedding_dims, filter_tuple, activation_value, dropout=0.25, channel1_weights=None, channel2_weights=None):
    channel1_input = Input(shape=(X_train_channel1.shape[1],), dtype='int32', name="channel1")
    channel2_input = Input(shape=(X_train_channel2.shape[1],), dtype='int32', name="channel2")
    channel1_x = Embedding(len(channel1_to_idx), embedding_dims, weights=channel1_weights, input_length=X_train_channel1.shape[1])(channel1_input)
    channel2_x = Embedding(len(channel2_to_idx), embedding_dims, weights=channel2_weights, input_length=X_train_channel2.shape[1])(channel2_input)
    n_layers = len(filter_tuple)
    cnn_nodes = []
    maxpool_list = []
    emb_dropout = Dropout(dropout)
    alpha_dropout = AlphaDropout(dropout)
    for i in range(n_layers):
        cnn = Convolution1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='valid', activation=activation_value, subsample_length=1)
        cnn_nodes.append(cnn)
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=(filter_tuple[i][0],))
        maxpool_list.append(maxpool)

    channel1_cnn = alpha_dropout(layers.Concatenate()([maxpool(cnn_node(emb_dropout(channel1_x))) for cnn_node, maxpool in zip(cnn_nodes, maxpool_list)]))
    channel2_cnn = alpha_dropout(layers.Concatenate()([maxpool(cnn_node(emb_dropout(channel2_x))) for cnn_node, maxpool in zip(cnn_nodes, maxpool_list)]))
    merged_layer = layers.Concatenate(name='merged')([channel1_cnn, channel2_cnn])
    # fully_connected = Dense(hidden_dense_dims, activation='relu')(x_dropout(merged_layer))
    # main_loss = Dense(len(authors_50), activation='softmax', name='main_output')(fully_connected)
    if no_of_labels == 2:
        main_loss = Dense(1, activation='sigmoid', name='main_output')(merged_layer)
    else:
        main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(merged_layer)
    model = Model(input=[channel1_input, channel2_input], output=main_loss)
    return model

def get_multichannel_model_shared_attn(no_of_labels, X_train_channel1, X_train_channel2, channel1_to_idx, channel2_to_idx, embedding_dims, filter_tuple, activation_value, dropout=0.25, channel1_weights=None, channel2_weights=None):
    channel1_input = Input(shape=(X_train_channel1.shape[1],), dtype='int32', name="channel1")
    channel2_input = Input(shape=(X_train_channel2.shape[1],), dtype='int32', name="channel2")
    channel1_x = Embedding(len(channel2_to_idx), embedding_dims, weights=channel1_weights, input_length=X_train_channel1.shape[1])(channel1_input)
    channel2_x = Embedding(len(channel1_to_idx), embedding_dims, weights=channel2_weights, input_length=X_train_channel2.shape[1])(channel2_input)
    n_filters = len(filter_tuple)
    cnn_nodes = []
    maxpool_list = []
    x_dropout = Dropout(dropout)
    for i in range(n_filters):
        cnn = Convolution1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='valid', activation=activation_value, subsample_length=1)
        cnn_nodes.append(cnn)
        # maxpool = AttLayer()
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=(filter_tuple[i][0],))
        maxpool_list.append(maxpool)

    channel1_cnn = layers.Concatenate()([maxpool(cnn_node(x_dropout(channel1_x))) for cnn_node, maxpool in zip(cnn_nodes, maxpool_list)])
    reshaped_channel1 = Reshape((1,1500))(channel1_cnn)

    channel2_cnn = layers.Concatenate([maxpool(cnn_node(x_dropout(channel2_x))) for cnn_node, maxpool in zip(cnn_nodes, maxpool_list)])
    reshaped_channel2 = Reshape((1,1500))(channel2_cnn)

    # merged_layer = merge([channel1_cnn, channel2_cnn], mode='concat', name='merged')
    # main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(merged_layer)

    # merged_layer = merge([reshaped_channel1, reshaped_channel2], mode='concat', concat_axis=1, name='merged')
    # merged_layer = merge([channel1_cnn, channel2_cnn], mode='concat', concat_axis=0, name='merged')
    merged_layer = layers.Concatenate(name='merged')([reshaped_channel1, reshaped_channel2])
    att_layer = AttLayer()(merged_layer)
    main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(att_layer)

    # #####remove, added just to check architecture
    # main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(merged_layer)
    model = Model(input=[channel1_input, channel2_input], output=main_loss)
    return model

def plot_learning_curve(history, opfpath):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    print(history.history.keys())
    # print history.history['acc']
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(opfpath)

def train_and_test(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, nb_epochs=100, batch_size=10, learning_rate=1e-4, model_type="", logdir=os.path.join(proj_basedir, "initial", "best_models"), bm_scores=True, binary=False):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpointer = ModelCheckpoint(save_best_only=True, monitor='val_acc', verbose=1, filepath=os.path.join(logdir, model_type+timestamp+".hdf5"), )
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    dict_callbacks = {'checkpointer':checkpointer, 'early':early_stopping}
    adam = Adam(lr=learning_rate)
    if binary:
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs, validation_data=(X_val, Y_val), callbacks=dict_callbacks.values())
    open(os.path.join(logdir, model_type+timestamp+".json"), 'w').write(model.to_json())

    model.summary()
    scores = model.evaluate(X_test, Y_test, batch_size=batch_size)
    plot_learning_curve(history, os.path.join(logdir, model_type+timestamp+".png"))


    # get best model scores
    if bm_scores:
        from keras.models import load_model
        best_model = load_model(os.path.join(logdir, model_type+timestamp+".hdf5"))
        bm_scores = best_model.evaluate(X_test, Y_test, batch_size=batch_size)
        # print("\n{}\tbest:{}\tlast: {}".format(best_model.metrics_names[1], scores[1], bm_scores[1]))
        print("\n{}\t{}\t{}".format(best_model.metrics_names[1], scores[1], bm_scores[1]))
        return scores[1], bm_scores[1]
    else:
        print("\n{}: {}".format(model.metrics_names[1], scores[1]))
        return scores[1]



def get_size_fold_from_processid(processid):
    sizes = [100, 200, 500, 1000]
    size = sizes[processid % len(sizes)]  # total no. of train tweets per author
    return size, processid // len(sizes)

def get_author_names_vauthors(author_lists_fpath, size):
    return [line.strip() for line in
            futils.get_lines_in_file(author_lists_fpath)]

def get_author_names_all_authors(author_lists_fpath):
    return [line.strip() for line in futils.get_lines_in_file(author_lists_fpath)]


def get_author_names_combined_expt(author_lists_fpath, size):
    return [line.strip() for line in futils.get_lines_in_file(author_lists_fpath)][:size]


# ds will not run in some cases without min_df
def get_data_for_nns(params):
    dataset, authors_count, _, _, _, _, sample_id, _, _ = params
    min_df = 0
    # dataset is already balanced and contains 5 sents per author for ds and amazon
    if dataset == "tweet":
        rootdir = os.path.join(os.path.join(resources_rootdir, "tweets"))
        authors = get_author_names_all_authors(os.path.join(resources_rootdir, "author_lists", "tweets.txt"))
        test_fold = 0
        # getting many samples of 10 from 1000 authors
        authors = authors[sample_id * authors_count:(sample_id + 1) * authors_count]
        print "size, test_fold: ", authors_count, test_fold
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_authors_train_test_koppel_vauthors(rootdir, authors, test_fold=test_fold, val_size=0.1)
    elif dataset == "ds":
        aa_rootdir = os.path.join(resources_rootdir, "posts")
        authors = [x.strip() for x in futils.get_lines_in_file_small(os.path.join(resources_rootdir, "author_lists", "posts.txt"))]
        authors = authors[sample_id * authors_count:(sample_id + 1) * authors_count]
        X_train, Y_train, _, X_test, Y_test, _ = get_data_for_simple_baseline_authorship_gender(aa_rootdir, authors)
        X_train, Y_train, X_val, Y_val, X_test, Y_test = convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size=0.1)
        min_df = 2
    elif dataset == "amazon":
        aa_rootdir = os.path.join(resources_rootdir, "reviews")
        authors = [x.strip() for x in futils.get_lines_in_file_small(os.path.join(resources_rootdir, "author_lists", "reviews.txt"))]
        authors = authors[sample_id * authors_count:(sample_id + 1) * authors_count]
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_authors_train_test_amazon(aa_rootdir, authors, val_size=0.1)
        min_df = 2
    else:
        raise ValueError('Dataset type does not exist')
    print "authors: ", authors
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, authors, min_df


# not working for dumped right now because no dumped for char
# and dumped for word does not give clue about which w2v to use for char
# this is the only problem
# infrastructure for dumped is ready in load_embs
def train_model_with_params(params, models_results_opdir):
    dataset, authors_count, model_type, feat_type, w2v_type, wiki_combined, sample_id, job_id, results_opdir = params
    details_for_fname = "_".join([str(x) for x in params if len(str(x)) < 15])

    X_train, Y_train, X_val, Y_val, X_test, Y_test, authors, min_df = get_data_for_nns(params)
    no_of_authors = len(authors)

    embedding_dims = 300
    filter_tuple = [[500,3],[500,4],[500,5]]
    # filter_tuple = [[100,2],[200,3],[300,4]]
    dropout = 0.25
    # activation_value = "relu"
    activation_value = "selu"
    nb_epochs = 100
    batch_size = 32
    learning_rate = 1e-4

    char_embedding_matrix = None
    embedding_matrix = None
    if model_type == "multichannel" or model_type == "attention":
        word_feats = Features("word", min_df=min_df)
        word_feats.fit(X_train)
        X_train_word, X_val_word, X_test_word, word_to_idx = word_feats.transform(X_train), word_feats.transform(X_val), word_feats.transform(X_test), word_feats.feature_map
        char_feats = Features("char_2", min_df=min_df)
        char_feats.fit(X_train)
        X_train_char, X_val_char, X_test_char, char_to_idx = char_feats.transform(X_train), char_feats.transform(X_val), char_feats.transform(X_test), char_feats.feature_map
        if w2v_type:
            if "both" in w2v_type:
                w2v_type = w2v_type[:-5]  # removing both
                char_embedding_matrix = create_embedding(char_to_idx, w2v_type, wiki_combined=wiki_combined)
            embedding_matrix = create_embedding(word_to_idx, w2v_type, wiki_combined=wiki_combined)
        futils.create_dumps_json(word_to_idx, os.path.join(models_results_opdir, "word_to_idx_{}.json".format(details_for_fname) ))
        futils.create_dumps_json(char_to_idx, os.path.join(models_results_opdir, "char_to_idx_{}.json".format(details_for_fname)) )

    if model_type == "multichannel":
        model = get_multichannel_model_shared(no_of_authors, X_train_char, X_train_word, char_to_idx, word_to_idx, embedding_dims, filter_tuple, activation_value, dropout, channel1_weights=char_embedding_matrix, channel2_weights=embedding_matrix)
        # model = get_multichannel_model(no_of_authors, X_train_char, X_train_word, char_to_idx, word_to_idx, embedding_dims, filter_tuple, activation_value, dropout, channel1_weights=char_embedding_matrix, channel2_weights=embedding_matrix)
        last_acc, bm_acc = train_and_test(model, [X_train_char, X_train_word], Y_train, [X_val_char, X_val_word], Y_val, [X_test_char, X_test_word], Y_test, nb_epochs, batch_size, learning_rate, details_for_fname, logdir=models_results_opdir)
    elif model_type == "attention":
        model = get_multichannel_model_shared_attn(no_of_authors, X_train_char, X_train_word, char_to_idx, word_to_idx, embedding_dims, filter_tuple, activation_value, dropout, channel1_weights=char_embedding_matrix, channel2_weights=embedding_matrix)
        last_acc, bm_acc = train_and_test(model, [X_train_char, X_train_word], Y_train, [X_val_char, X_val_word], Y_val, [X_test_char, X_test_word], Y_test, nb_epochs, batch_size, learning_rate, details_for_fname, logdir=models_results_opdir)
    elif model_type == "singlechannel":
        single_feats = Features(feat_type, min_df=min_df)
        single_feats.fit(X_train)
        X_train_single, X_val_single, X_test_single, single_to_idx = single_feats.transform(X_train), single_feats.transform(X_val), single_feats.transform(X_test), single_feats.feature_map
        futils.create_dumps_json(single_to_idx, os.path.join(models_results_opdir, "_to_idx_{}.json".format(details_for_fname)) )
        if w2v_type:
            embedding_matrix = create_embedding(single_to_idx, w2v_type, wiki_combined=wiki_combined)
        model = get_single_channel_model(no_of_authors, X_train_single, embedding_dims, single_to_idx, filter_tuple, dropout, activation_value, weights=embedding_matrix)
        last_acc, bm_acc = train_and_test(model, X_train_single, Y_train, X_val_single, Y_val, X_test_single, Y_test, nb_epochs, batch_size, learning_rate, details_for_fname, logdir=models_results_opdir)

    with open(os.path.join(models_results_opdir, "authors_{}.txt".format(details_for_fname)), "w") as fhandle:
        fhandle.write("\n".join(authors)+"\n")
    with open(os.path.join(models_results_opdir, "results_{}.txt".format(details_for_fname)), "w") as fhandle:
        fhandle.write("authors: {}\n".format(len(authors)))
        fhandle.write("last acc: {}\n".format(last_acc))
        fhandle.write("best model acc: {}\n".format(bm_acc))


## NOT TESTED: 11/21 1:00p
def create_results_models_opdir(params):
    dataset, authors_count, model_type, _, w2v_type, wiki_combined, sample_id, job_id, results_opdir = params
    details_for_fname = "_".join([str(x) for x in params if len(str(x)) < 15])
    models_results_opdir = "{}_{}".format(os.path.join(results_opdir, details_for_fname),
                                          time.strftime("%Y%m%d-%H%M%S"))
    print models_results_opdir
    futils.create_opdir(models_results_opdir)
    shutil.copy(os.path.realpath(__file__), models_results_opdir)
    return models_results_opdir



# THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32' python -m main_all.all_models ds 50 multichannel "" ds_fasttext_bin
# THEANO_FLAGS='mode=FAST_RUN,device=gpu0,floatX=float32' python -m main_all.all_models ds 50 multichannel "" ds_fasttext_bin "" 0
# THEANO_FLAGS='mode=FAST_RUN,device=gpu1,floatX=float32' python -m main_all.all_models ds 50 singlechannel char_2 "" "" 0
# THEANO_FLAGS='mode=FAST_RUN,device=gpu1,floatX=float32' python -m main_all.all_models ds 50 singlechannel char_2
#     dataset, authors_count, model_type, feat_type, w2v_type, wiki_combined, sample_id, job_id, results_opdir
if __name__ == "__main__":
    logdir = os.path.join(resources_rootdir, "results_models")
    params = futils.get_all_params(sys.argv, defaults=["tweets", 10, "singlechannel", "char_2", "", True, 0, "", logdir],
                                   var_names=["dataset", "authors_count", "model_type", "feat_type", "w2v_type", "wiki_combined", "sample_id", "job_id", "results_opdir"])   # params used for fname later
    models_results_opdir = create_results_models_opdir(params)
    train_model_with_params(params, models_results_opdir)







