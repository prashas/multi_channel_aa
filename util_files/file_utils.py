import io
import os
from pickle import load, dump
import json
import re
from shutil import copyfile

# from BeautifulSoup import BeautifulSoup
from bs4 import BeautifulSoup
# import regex
import warnings
from config import resources_rootdir

__author__ = 'shrprasha'

def get_files_in_folder(folder, extension=None, fname_contains=None):
    all_files = sorted(next(os.walk(folder))[2])
    if ".DS_Store" in all_files:
        all_files.remove(".DS_Store")
    if extension:
        extension = "." + extension
        for fname in all_files:
            if extension not in fname:
                all_files.remove(fname)
    if fname_contains:
         for fname in all_files:
            if fname_contains not in fname:
                all_files.remove(fname)
    return all_files

def get_subfolders(folder):
    return sorted(next(os.walk(folder))[1])

def load_dumps(fpath):
    with open(fpath, "r") as fhandle:
        obj = load(fhandle)
    return obj

def create_dumps(obj, fpath):
    with open(fpath, "w") as fhandle:
        dump(obj, fhandle)

def load_dumps_json(fpath):
    with open(fpath, "r") as fhandle:
        obj = json.load(fhandle)
    return obj

def create_dumps_json(obj, fpath):
    with open(fpath, "w") as fhandle:
        json.dump(obj, fhandle)

def create_opdir(opdir):
    if not os.path.exists(opdir):
        os.makedirs(opdir)

def get_lines_in_file_small(fpath, remove_empty=True, encoding='utf-8'):
    with io.open(fpath, encoding=encoding) as fhandle:
        lines = fhandle.readlines()
    cleaned = []
    if remove_empty:
        for line in lines:
            if line.strip():
                cleaned.append(line)
        return cleaned
    return lines

def get_whole_file_content(fpath, encoding='utf-8'):
    with io.open(fpath, "r", encoding=encoding) as fhandle:
        return fhandle.read()

def get_lines_in_file(fpath):
    """
    Uses a generator to read a large file lazily
    """
    with io.open(fpath, encoding='utf-8') as fhandle:
        while True:
            data = fhandle.readline()
            if not data:
                break
            yield data

def write_list_to_file(l1, fpath, strip=False):
    # with io.open(fpath, "w", encoding='utf-8') as fhandle:
    with io.open(fpath, "w") as fhandle:
        for i in l1:
            if strip:
                i = i.strip()
            fhandle.write(i + u"\n")

def print_obj(obj):
    attrs = vars(obj)
    print(', '.join("%s: %s" % item for item in list(attrs.items())))


# use BeautifulSoup to remove html tags; cleans up all well-formed html tags
def remove_tags_with_bs(raw_text):
    try:
        clean_text = BeautifulSoup(raw_text).text
    except UserWarning:
        pass
    return clean_text

# corpus has ill-formed html where one tag can occur inside another
# Eg, in corpus2/file4 -> <img src= "<a href...></a>" />
# simple regex to catch anything between <> does not work
# function uses recursive regex to catch the innermost <> first
# regex adapted from http://stackoverflow.com/questions/26385984/recursive-pattern-in-regex
def remove_html_tags(raw_text):
    return re.sub("<((?>[^<>]+|(?R))*)>", "", raw_text)

def remove_urls(raw_text, replace_with='<URL>'):
    # regex from http://www.rubular.com/r/A5jxmCDDaw
    clean_text = re.sub(r"(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})", replace_with, raw_text, flags=re.MULTILINE)
    return clean_text

# call all clean and normalization functions here such as remove_urls
def clean_and_normalize_text(raw_text):
    # html_tags_removed_text = remove_urls(remove_tags_with_bs(remove_html_tags(raw_text)))
    html_tags_removed_text = remove_urls(remove_tags_with_bs(raw_text))
    return html_tags_removed_text

def replace_tweet_usernames(raw_text, replace_with='<USERNAME>'):
    clean_text = raw_text
    # clean_text = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", '@USERNAME', raw_text)
    clean_text = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([_A-Za-z0-9]+)", replace_with, raw_text)
    return clean_text

def replace_tweet_numbers(raw_text, replace_with='<NUMBER>'):
    clean_text = re.sub(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)*", replace_with, raw_text)
    return clean_text

def replace_tweet_date_time(raw_text, replace_with='<DATETIME>'):
    clean_text = re.sub(r"(\d+[/-]\d+[/-]\d+)", replace_with, raw_text)
    clean_text = re.sub(r"\d{1,2}((am|pm)|(:\d{1,2})(am|pm)?)", replace_with, clean_text, flags=re.IGNORECASE)
    return clean_text

# call all clean and normalization functions here such as remove_urls
def clean_and_normalize_tweet(raw_text):
    # html_tags_removed_text = remove_urls(remove_tags_with_bs(remove_html_tags(raw_text)))
    raw_text = raw_text.lower()  # lowered here before any other preprocessing because remove urls might add uppercase @URL and replace_tweet_usernames might add uppecase @USERNAME
    html_tags_removed_text = replace_tweet_usernames(remove_urls(remove_tags_with_bs(raw_text)))
    return html_tags_removed_text.strip()

"""added later"""
def replace_replacers_with_nums(raw_text):
    raw_text = raw_text.replace('<URL>', '1')
    raw_text = raw_text.replace('<USERNAME>', '2')
    raw_text = raw_text.replace('<DATETIME>', '3')
    raw_text = raw_text.replace('<NUMBER>', '4')
    return raw_text

def clean_and_normalize_tweet_koppel(raw_text, lowercase=True):
    # html_tags_removed_text = remove_urls(remove_tags_with_bs(remove_html_tags(raw_text)))
    if lowercase:
        raw_text = raw_text.lower()  # lowered here before any other preprocessing because remove urls might add uppercase @URL and replace_tweet_usernames might add uppecase @USERNAME
    html_tags_removed_text = replace_tweet_numbers(replace_tweet_date_time(replace_tweet_usernames(remove_urls(remove_tags_with_bs(raw_text)))))
    html_tags_removed_text = replace_replacers_with_nums(html_tags_removed_text)
    # html_tags_removed_text = replace_tweet_numbers(replace_tweet_date_time(replace_tweet_usernames(remove_urls(remove_tags_with_bs(raw_text), "<URL>"), "<USERNAME>"), "<DATETIME>"), "<NUMBER>")
    return html_tags_removed_text.strip()



def write_to_file(fpath, text, encoding="latin1"):
    with io.open(fpath, "w", encoding=encoding) as fhandle:
        try:
            fhandle.write(text)
        except:
            print "=" * 40
            print(text)
            print "=" * 40



# mandatory first and then optional ones
# if empty optional one, give as ""
# "" for False bool arg: since all else when converted to bool gives True
def get_all_params(args, defaults, var_names=None):
    print "args: ", args[1:]
    print defaults
    params = defaults[:]
    # 10/24 6:00p not tested on all
    for idx, arg in enumerate(args[1:]):
        if isinstance(params[idx], bool):
            if arg is not None:  # for bool args: to get False we need to provide empty since bool("False")=True
                params[idx] = bool(arg)
        elif arg:  # if not None used for these, won't take default params
            if isinstance(params[idx], int):
                params[idx] = int(arg)
            else:
                params[idx] = arg
    print "params: ", params
    if var_names:
        print zip(var_names, params)
    return params if len(params) > 1 else params[0]   # otherwise gives error for single param


if __name__ == "__main__":
    print clean_and_normalize_tweet_koppel("This is a test @username 1. 1.3 190 0.1 2012-3-4 http://bitly.cc bitly.cc www.bitly.cc")
    # print clean_and_normalize_tweet_koppel("@562citylife no.. that would be old skool lol")
    # print(clean_and_normalize_tweet_koppel("@_missnicole and a spa ;)"))

    # get_all_params(["0", 0], [True])





