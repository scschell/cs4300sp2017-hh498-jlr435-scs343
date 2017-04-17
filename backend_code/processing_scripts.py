#!/usr/bin/env python
""" Contains back-end scripts needed for processing raw text files within a directory.
Author @ Sayge Schell <scs343@cornell.edu>
"""

from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import re
import os

"""Creates a key for each book in the file directory, given a file path.
    Input: file path as string
    Ouput: key as string """
def nice_key(file_path):
    k = file_path.rpartition("/")
    key = k[2].partition(".txt")
    return key[0]

"""Takes a filename and returns its extracted title, author, release date, and script as a tuple.
   Input: .txt filename
   Ouput: (title, author, release date, script) tuple of strings
"""
def process_raw_text_files(filename):
    transcript_filename = filename
    title = ""
    author = ""
    year = ""
    text = ""
    script = False   #True when we are within the book's text, False otherwise
    in_title = False #This allows us to save titles that span multiple lines in the file
    
    with open(transcript_filename) as f:
        for line in f:
            #If we are within the text of the book
            if script:
                #If we have reached the end of the book's text, stop
                if "End of the Project Gutenberg EBook" in line:
                    break
                #Otherwise, save the line
                else:
                    text = text + str(line)
            #If we found the title
            if "Title:" in line and title == "":
                in_title = True
                title = line[7:]
                title = title.rstrip()
            #If we found the author
            elif "Author:" in line and author == "":
                in_title = False
                author = line[8:]
                author = author.rstrip()
            #If we found the release date
            elif "Release Date:" in line and year == "":
                in_title = False
                y = line[14:]
                y = re.sub(r'\[.*?\]', '', y)
                year = re.findall(r"\D(\d{4})\D", y)
                year = year[0]
            #If we found where the body of the book begins
            elif "***" in line and text == "":
                script = True
                in_title = False
            #If none of these are true and we're still in the title, title spans multiple lines
            elif in_title:
                title = title + " " + str(line).lstrip()
                title = title.rstrip()
    
    return (title, author, year, text.replace("_", " "))

""" Builds the data dictionary given a path.
    Input: path or directory where files are
    Output: a dictionary containing all book data"""
def build_dicts(path):
    data = []
    i = 0
    #For each file in each directory
    for directory, directories, filenames in os.walk(path):
        for filename in filenames:
            #if they are .html, parse them and add them to dicts
            if filename.endswith(".txt"):
                entry = {}
                url = os.path.join(directory, filename)
                title, author, year, text = process_raw_text_files(url)
                entry["title"] = title
                entry["author"] = author
                entry["year"] = year
                s = text.replace('\n', ' ').replace('\r', '')
                entry['script'] = s.decode('unicode_escape')
                entry["book_id"] = nice_key(url)
                data.append(entry)

    return data

"""Builds supplemental useful dictionaries that relate titles, ids, and indexes.
    Input: list of dictionaries containing book data
    Ouput: 5 lists relating titles, ids, and indexes.
"""
def build_supp_dicts(data):
    #Create dictionaries that relate titles, indexes, and ids
    book_id_to_index = {book_id:index for index, book_id in enumerate([d['book_id'] for d in data])}
    book_title_to_id = {}
    for title, bid in zip([d['title'] for d in data], [d['book_id'] for d in data]):
        if title in book_title_to_id and bid.partition("-")[1] < book_title_to_id[title].partition("-")[1]:
            continue
        else:
            book_title_to_id[title] = bid

    book_id_to_title = {v:k for k,v in book_title_to_id.iteritems()}
    book_title_to_index = {name:book_id_to_index[book_title_to_id[name]] for name in [d['title'] for d in data]}
    book_index_to_title = {v:k for k,v in book_title_to_index.iteritems()}

    #If we have multiple copies of the same book, delete them from the inde
    if len(book_id_to_index) != len(book_index_to_title):
        todel = []
        for bid in book_id_to_index:
            if bid not in book_title_to_id.values():
                todel.append(bid)
        for item in todel:
            del book_id_to_index[item]
    return book_id_to_index, book_title_to_id, book_id_to_title, book_title_to_index, book_index_to_title

"""Build tf-idf vectors of script data.
    Input: list of dictionaries containing book data
    Output: index to vocab list, book by vocab vector array
"""
def build_vectors(data):
    n_feats = 200000
    book_by_vocab = np.empty([len(data), n_feats])
    tfidf_vec = TfidfVectorizer(input=data, min_df=10, max_df=.8, max_features=200000, stop_words=None, norm='l2')
    #extract scripts
    scripts = [item['script'] for item in data]
    #call fit_transform() on list of scripts in data
    book_by_vocab = tfidf_vec.fit_transform(scripts)

    # Construct a inverted map from feature index to feature value (word) for later use
    index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}

    book_by_vocab = book_by_vocab.toarray()
    return index_to_vocab, book_by_vocab

def main():
	data = build_dicts('data')
	book_id_to_index, book_title_to_id, book_id_to_title, book_title_to_index, book_index_to_title = build_supp_dicts(data)
	index_to_vocab, book_by_vocab = build_vectors(data)
        print(book_by_vocab)

main()
