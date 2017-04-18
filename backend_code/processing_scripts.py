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

""" Finds cosine similarity between two books using their transcripts.
    Input:
        b1: The title of the first book we are looking for.
        b2: The title of the second book we are looking for.
        book_title_to_index : dict of book titles mapped to their indices
        book_by_vocab : matrix of books and their vocabularies
    Ouput:
        similarity: Cosine similarity of the two book transcripts.
"""
def get_sim(b1, b2, book_title_to_index, book_by_vocab):
    index1 = book_title_to_index[b1]
    index2 = book_title_to_index[b2]
    #compute and return dot product; vectors are already normalized
    return np.dot(book_by_vocab[index1], book_by_vocab[index2])

"""Builds book similarity vector
    Input:
        book_by_vocab: matrix of books and their vocbaularies
        data: all data pertaining to books
        book_index_to_title: dict mapping book indices to titles
        book_title_to_index: dict mapping book titles to indices
    Ouput: book similarity vector"""
def build_similarities(book_by_vocab, data, book_index_to_title, book_title_to_index):
    book_sims = np.empty([len(data), len(data)], dtype = np.float32)
    #For each pair of items
    for i in range(0, len(book_by_vocab)):
        for j in range(0, len(book_by_vocab)):
            if i not in book_index_to_title or j not in book_index_to_title:
                book_sims[i][j] = -1
            #If it is the diagonal, set to 0
            elif i == j:
                book_sims[i][j] = 0
            #Otherwise, find the similarity
            else :
                book_sims[i][j] = get_sim(book_index_to_title[i], book_index_to_title[j], book_title_to_index, book_by_vocab)
    return book_sims

"""Finds top and bottom x of books similar to a given book title.
    Input:
        title: book title
        x: number of highest/lowest books to display
    Ouput: printed ordered list of top entries, ordered list of bottom entries"""
def find_top_and_bottom_sims(title, x):
    #Find index for Lucky Pehr
    index = book_title_to_index[title]
    #Find x most and least similar books
    highest_indices = (-book_sims[index]).argsort()[:x]
    
    print("MOST SIMILAR:")
    i = 1
    for ind in highest_indices:
        print(str(i) + ". ", round(book_sims[index][ind], 2), book_index_to_title[ind])
        i = i + 1
    
    #Find x least similar books
    lowest_ind = (book_sims[index]).argsort()
    lowest_ind = [item for item in lowest_ind if book_sims[index][item] >= 0]
    to_del = np.argwhere(lowest_ind==index)
    if index in lowest_ind:
        lowest_ind.remove(index)
    lowest_indices = np.delete(lowest_ind, to_del)[:x]
    
    print("\nLEAST SIMILAR:")
    i = 0
    for ind in lowest_indices:
        print(str(i) + ". ", round(book_sims[index][ind], 2), book_index_to_title[ind])
        i = i + 1

""" Creates a JSON object storing the cosine similaritites of the books. """
def create_json_matrix(item, name):
    with open(name, 'w') as f:
        json.dump(item.tolist(), f, ensure_ascii=False)

def load_json_matrix(filepath):
    with open(filepath, "rb") as f:
        i = json.load(f)
        item = np.array(i)
    return item

def create_json(item, name):
    with open(name, 'w') as fp:
        json.dump(item, fp, ensure_ascii=False)

def load_json(filepath):
    with open(filepath, "r") as fp:
        item = json.load(fp)
    return item

def main():
	data = build_dicts('../data')
	book_id_to_index, book_title_to_id, book_id_to_title, book_title_to_index, book_index_to_title = build_supp_dicts(data)
	index_to_vocab, book_by_vocab = build_vectors(data)
	book_sims = build_similarities(book_by_vocab, data, book_index_to_title, book_title_to_index)
        create_json_matrix(book_sims, "book_sims.json")
        #book_sims = load_json_matrix("book_sims.json") to load the book

main()
