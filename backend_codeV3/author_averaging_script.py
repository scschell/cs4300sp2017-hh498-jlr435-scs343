#!/usr/bin/env python
""" Script by Sayge Schell <scs343@cornell.edu>
"""

import cPickle as pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from collections import defaultdict

def create_pickle(item, name):
    with open(name, 'wb') as f:
        pickle.dump(item, f)

def load_pickle(filepath):
    with open(filepath,'rb') as f:
        item = pickle.load(f)
    return item

def merge_dicts(*dict_args):
        """
        From: 
        http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
        """
        result = {}
        for dictionary in dict_args:
                result.update(dictionary)
        return result

def gen_author_to_books(title_to_authors):
    author_to_titles = defaultdict(list)
    #Check through all titles
    for title in title_to_authors:
        authors = title_to_authors[title]
        #for each author
        for author in authors:
            #if it doen't exist in the dictionary, add it
            if author not in author_to_titles:
               author_to_titles[author] = []
            #then, add the book under that author
            author_to_titles[author].append(title)
    return author_to_titles

def gen_average_author_vecs(author_to_titles, book_vectors, title_to_index):
    author_vecs = np.zeros((len(author_to_titles), len(book_vectors[0])))
    index_to_author = defaultdict(list)
    author_to_index = defaultdict(list)

    #Compute avg tfidf vector for each author
    i = 0
    #Compute for each author
    for author in author_to_titles:
        
        #1/|titles written by author|
        vector = np.zeros(len(book_vectors[0]))
        denom = 1 / float(len(author_to_titles[author]))
        #Sum up all vectors written by the author
        for title in author_to_titles[author]:
	    query = book_vectors[title_to_index[title]]
            vector = vector + query
        #avg vector = 1/|books written by author| * sum(vectors of books written by author)
        average_vector = denom * vector
        if average_vector.any() > 0:
            #store and norm
            author_vecs[i] = average_vector / (np.linalg.norm(average_vector))
            #store index for later use
            index_to_author[i] = author
	    author_to_index[author] = i
            i = i + 1
	
    return author_to_index, index_to_author, author_vecs

"""
Unifies title data between the two title dictionaries.
"""
def remove_non_overlaps(title_to_authors, title_to_index):
    to_del = []
    to_del2 = []

    #check for non-overlaps in one dict
    for title in title_to_authors:
        if title not in title_to_index:
            to_del.append(title)
    for item in to_del:
        del title_to_authors[item]

    #check in the other dict
    for title in title_to_index:
        if title not in title_to_authors:
            to_del2.append(title)
    for item in to_del2:
        del title_to_index[item]

    return title_to_authors, title_to_index

def main():
    #load necessary pickle files
    print("LOADING FILES")
    book_vectors = load_pickle('book_vectors.pickle') 
    title_to_index = load_pickle('book_title_to_index.pickle')
    index_to_title = load_pickle('book_index_to_title.pickle')
    authors1 = load_pickle('data1_authors.pickle')
    authors2 = load_pickle('data2_authors.pickle')
    authors3 = load_pickle('data3_authors.pickle')
    print("MERGING DATA")
    title_to_authors = merge_dicts(authors1, authors2, authors3)

    #generate dict of author : book titles
    print("GENERATING NEW DICTS")
    title_to_authors, title_to_index = remove_non_overlaps(title_to_authors, title_to_index)
    author_to_titles = gen_author_to_books(title_to_authors)

    #generate average vector for each author
    print("GENERATING AVERAGE VECTORS FOR EACH AUTHOR")
    author_to_index, index_to_author, author_vectors = gen_average_author_vecs(author_to_titles, book_vectors, title_to_index)

    #save vectors
    print("SAVING PICKLES")
    create_pickle(title_to_authors, "title_to_authors.pickle")
    create_pickle(author_to_titles, "author_to_titles.pickle")
    create_pickle(author_vectors, "author_vectors.pickle")
    create_pickle(author_to_index, "author_to_index.pickle")
    create_pickle(index_to_author, "index_to_author.pickle")

main()
