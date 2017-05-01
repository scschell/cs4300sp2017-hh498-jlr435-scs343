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

def rocchio(query, relevant, irrelevant, a=.3, b=.3, c=.8, clip = True):
    '''
    Arguments:
        query: a string representing the name of the movie being queried for
        
        relevant: a list of vectors of relevant authors/books
        
        irrelevant: a list of query vectors of irrelevent authors/books 

	title_to_index: a dictionary mapping all book titles to indices

	book_vectors: the matrix containing all book information

        author_vectors: the matrix containg all author information
        
        a,b,c: floats, corresponding to the weighting of the original query, relevant queries,
        and irrelevant queries, respectively.
        
        clip: boolean, whether or not to clip all returned negative values to 0
        
    Returns:
        q_mod: a vector representing the modified query vector. this vector should have no negatve
        weights in it when clip is True.
        
    Note: you will have to access the original tfidf matrix "doc_by_vocab", along with
    other data structures you created previously in the assignment. Be sure to handle the
    cases where relevant and irrelevant are empty lists.
    '''
    # Code completion 5.1
    #q1 = a * q0 + b * (1/|Dr|)*sum(relevant d) - c*(1/|Dnr|)*sum(not relevant d)
    
    #Part 1 of equation ( a * q0 )
    q0 = query
    part1 = a * q0
    
    #Part 2 of equation ( b * 1/|Dr| * sum (relevant d))
    #if there are no relevant queries, set part2 to 0
    if len(relevant) == 0:
        part2 = 0
    #otherwise, calculate 2nd part of the equation
    else:
        part2a = b * (1/float((len(relevant))))
        part2b = np.zeros(len(q0))
        #for each relevant query, sum up the vectors
        for item in relevant:
            part2b = part2b + item
        part2 = part2a * part2b
    
    #Part 3 of equation ( c * 1/|Dnr| * sum (not relevant d))
    #if there are no irrelevant queries, set part3 to 0
    if len(irrelevant) == 0:
        part3 = 0
    #otherwise, calculate 3rd part of the equation
    else:
        part3a = c * (1/float((len(irrelevant))))
        part3b = np.zeros(len(q0))
        #for each irrelevant query, sum up the vectors
        for item in irrelevant:
            part3b = part3b + item
        part3 = part3a * part3b
    
    #Combine all parts and return
    q1 = part1 + part2 - part3
    #clip if needed
    if clip:
        q1 = np.ndarray.clip(q1, 0)
    return q1

def main():
    #load book vectors
    print("LOADING PICKLE FILES")
    book_vectors = load_pickle('book_vectors.pickle')
    title_to_index = load_pickle('book_title_to_index.pickle')
    index_to_title = load_pickle('book_index_to_title.pickle')
    index_to_author = load_pickle('index_to_author.pickle')
    author_vectors = load_pickle('author_vectors.pickle')

    #Example of how to use
    """
    index = 435
    i2 = 444
    a1 = 765

    n1 = index_to_title[435]
    n2 = index_to_title[444]
    an1 = index_to_author[765]

    query = book_vectors[index]
    relevant = [book_vectors[i2], author_vectors[a1]]
    
    new_q = rocchio(query, relevant, [], a=.3, b=.3, c=.8, clip = True)
    """


main()
