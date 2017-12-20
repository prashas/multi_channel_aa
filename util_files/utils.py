# -*- coding: utf-8 -*-
from itertools import dropwhile
import nltk
# from nltk.tokenize.punkt import PunktWordTokenizer
# import constants
import re
import numpy
import string
from scipy import spatial
from collections import Counter
from nltk.tag import StanfordNERTagger
from util_files import twokenize


def ctr_remove_below_threshold(ctr, threshold):
    for key, count in reversed(ctr.most_common()):
        if count < threshold:
            del ctr[key]
        else:
            break


def ark_tweet_tokenizer(text):
    return twokenize.tokenizeRawTweetText(text)


def get_ngrams_from_list(l1, n_start, n_end):
    # print "no of words", len(l1)
    if n_start > 1:
        pos_ngrams = []
    else:
        pos_ngrams = l1[:]
        n_start = 2
    for i in range(n_start, n_end + 1):
        pos_ngrams.extend(nltk.ngrams(l1, i))
    # return Counter(pos_ngrams)
    return pos_ngrams

def get_ngrams_with_step_size(l1, n, step_size):
    all_ngrams = nltk.ngrams(l1, n)
    if n == 1:
        return list(all_ngrams)
    return [ngram for i, ngram in enumerate(all_ngrams) if i%step_size==0]


def load_ner():
    st = StanfordNERTagger(
        # '/Users/shrprasha/Downloads/Softwares/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz',
        # # "/Users/shrprasha/Downloads/Softwares/stanford-english-corenlp-2015-12-11-models.jar",
        # '/Users/shrprasha/Downloads/Softwares/stanford-ner-2015-12-09/stanford-ner.jar'
        "/Users/shrprasha/Downloads/Softwares/stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz",
        "/Users/shrprasha/Downloads/Softwares/stanford-ner-2015-04-20/stanford-ner.jar",
    )
    return st


def char_ngrams(text, n):
    text_ngrams = []
    for i in range(0, len(text) - n + 1):
        text_ngrams.append(text[i:n + i])
    return text_ngrams


def char_ngrams_range(text, n_start, n_end):
    text_ngrams = []
    for n in range(n_start, n_end):
        for i in range(0, len(text) - n + 1):
            text_ngrams.append(text[i:n + i])
    return text_ngrams


def get_words_in_text(text):
    words = []
    # line = remove_punct(text)
    sentence_list = get_sentences(text)
    for sentence in sentence_list:
        word_list = get_words(sentence)
        words.extend(word_list)
    return words


def get_sentences(document):
    sentences = nltk.sent_tokenize(document)
    return sentences


# used everywhere
def get_words(sentence):
    return nltk.word_tokenize(sentence)
    #return nltk.word_tokenize(remove_punct(sentence)) #do not uncomment this; call remove_punct and give output to get_words


def create_n_gram_profile(words, n):
    return nltk.ngrams(words, n)


def create_n_gram_profile_n(words, n):
    return nltk.ngrams(words, n)


def remove_punct(s):
    punctuation = re.compile(r'[-.?!,":;()`]')
    #    bullet = re.compile(ur'•', re.UNICODE)
    bullet = re.compile(ur'\u2022')
    #    bullet = re.compile(r'u\'\u2022\'')
    #    bullet = re.compile("•")
    return bullet.sub(" ", punctuation.sub(" ", s))


def remove_punct_from_list(l):
    puncts = string.punctuation + u'\u2022'
    no_puncts = [x for x in l if x not in puncts]
    return no_puncts


def get_words_in_line(line):
    words = []
    #    line = remove_punct(line)
    sentence_list = get_sentences(line)
    for sentence in sentence_list:
        word_list = get_words(sentence)
        sw_in_sent = [w.lower() for w in
                      word_list]  # just changing this to take all words wont change a thing because stopwords have been used throughout the project
        words.extend(sw_in_sent)
    return words


def common_between_lists(susp_nes, src_nes):
    return set(susp_nes).intersection(set(src_nes))


def compare_tuples(t1, t2, match):
    if (len(set(t1).intersection(set(t2))) >= match):
        #        print "yes"
        return True
    else:
        #        print "no"
        return False


def merge_lists_remove_duplicates(l1, l2):
    #did not work since list is a list of lists returns unhashable type:list
    #    return sorted(l1 + list(set(l2) - set(l1)))
    for x in l1:
        if x not in l2:
            l2.append(x)
    return l2


def allwords(document):
    document.seek(0)
    doc_all = []
    for line in document:
        doc_all.extend(get_words_in_line(line))
    return doc_all


def total_words(document):
    s = 0
    a = 0
    #    for line in document:
    #        sentence_list = get_sentences(line)
    #        for sentence in sentence_list:
    #            word_list = get_words(sentence)
    #            for w in word_list:
    #                if w.lower() in constants.stopwords:
    #                    s+=1
    #                a+=1
    for line in document:
        a += len(get_words_in_line(line))


#    print a
#    print s

#used by get_avg_distance_betn_plag_seg_and_other_segs        
def get_distance_betn_two_vectors(t1, t2):
    #    print "t1: ", t1
    #    print "t2: ", t2
    fv1 = numpy.array(t1)
    fv2 = numpy.array(t2)
    return numpy.linalg.norm(fv1 - fv2)


def get_cosine_distance(t1, t2):
    return spatial.distance.cosine(t1, t2)


def make_normalize(omin, omax):
    def normalize(x):
        #        b = 1
        #        a = 0
        #        return a + (x-omin) * (b - a) / (omin - omax)
        return 1.0 if (omin == omax) else (x - omin) / (omax - omin)

    return normalize


def print_lol(list_of_lists):
    print "["
    for list in list_of_lists:
        print list, ","
    print "]"


def remove_multiple_whitespaces(passage):
    passage = passage.replace(u'\xa0', u' ')
    return re.sub('\s{2,}', ' ', passage).strip()


def display_dict_sorted_keys(mydict):
    for key in sorted(mydict.iterkeys()):
        print "%s: %s" % (key, mydict[key])
    print


def get_sil_coeff(dist_own, dist_other):
    return float(dist_other - dist_own) / max(dist_other, dist_own)


#input: list of words in a sentence
def get_pos_tags_hunpos(words_in_sent, ht):
    #tagger given as input since it will be slow to load the model each time
    #ht = HunposTagger('/home/prasha/Documents/softwares/hunpos/english.model', encoding='utf-8')
    tags = ht.tag(get_words(words_in_sent))
    return tags




    


