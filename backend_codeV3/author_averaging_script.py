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



def rocchio(query, relevant, irrelevant, title_to_index, book_vectors, a=.3, b=.3, c=.8, clip = True):
    '''
    Arguments:
        query: a string representing the name of the movie being queried for
        
        relevant: a list of strings representing the names of relevant movies for query
        
        irrelevant: a list of strings representing the names of irrelevant movies for query

	title_to_index: a dictionary mapping all book titles to indices

	book_vectors: the matrix containing all book information
        
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
    q0 = book_vectors[title_to_index[query]]
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
            part2b = part2b + book_vectors[title_to_index[item]]
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
            part3b = part3b + book_vectors[title_to_index[item]]
        part3 = part3a * part3b
    
    #Combine all parts and return
    q1 = part1 + part2 - part3
    #clip if needed
    if clip:
        q1 = np.ndarray.clip(q1, 0)
    return q1

def mean_average_precision_rocchio(relevant_in, irrelevant_in, title_to_index, book_vectors, index_to_title):
    '''
    Arguments:
        relevant_in: a list of (query, [relevant documents]) pairs, representing the queries we
        want to evaluate our ranking system against,along with some relevant documents.
        
        irrelevant_in: a list of (query, [irrelevant documents]) pairs, representing the queries we
        want to evaluate our ranking system against, along with some irrelevant documents.
        
        You can assume that relevant_in[i][0] = irrelevant_in[i][0] (i.e. the queries are in the
        same order). You can use the default rocchio parameters.
    Returns:
        mean_average_precision: float corresponding to the average AP statistic for the input queries
        and the similarity matrix
    '''
    # Code completion 5.2
    modified_vectors = []
    new_similarities = {}
    new_sim_books = {}
    precisions = []
    
    #Part 1 : Create modified vectors
    for i in range(0, len(relevant_in)):
        modified_vectors.append(rocchio(relevant_in[i][0], relevant_in[i][1], irrelevant_in[i][1], title_to_index, book_vectors))
    
    #Part 2 : Find the new top similar movies for modded vectors using cosine similarity
    #For each modded vector
    for j in range(0, len(modified_vectors)):
        new_similarity = np.zeros(len(book_vectors))
        #Compare to each other movie
        for k in range(0, len(book_vectors)):
            #If we are comparing it to itself, assign similarity to 0
            if k == title_to_index[relevant_in[j][0]]:
                new_similarity[k] = 0
            #Otherwise, find the similarity
            else :
                new_similarity[k] = np.dot(modified_vectors[j], book_vectors[k])
        #Save the similarity
        new_similarities[relevant_in[j][0]] = new_similarity
        
        #Sort the movies
        highest_indices = (-new_similarities[relevant_in[j][0]]).argsort()
        to_del = np.argwhere(highest_indices==title_to_index[relevant_in[j][0]])
        indices = np.delete(highest_indices, to_del)
        #Put ordered list in dictionary
        for ind in indices:
            if relevant_in[j][0] not in new_sim_books:
                new_sim_books[relevant_in[j][0]] = []
            new_sim_books[relevant_in[j][0]].append(index_to_title[ind]) 
        
	"""
        #Print out the top 10
        top_ten = indices[:10]
        print(relevant_in[j][0], "---")
        for ind in top_ten:
            print(movie_index_to_name[ind])
        print("\n")
	"""
        
    #Part 3 : Find precisions
    for l in range(0, len(relevant_in)):
        #give precision function new list of top movies and pre-determined relevant movies
        precisions.append(average_precision(new_sim_books[relevant_in[l][0]], relevant_in[l][1]))
    #sum up all items in the array and return the average
    return sum(precisions)/float(len(precisions))

def average_precision(ranking_in, relevant):
    '''
    Arguments:
        ranking_in: sorted ranking of movies, starting with the most most similar, and ending
        with the least similar.
        
        relevant: iterable of movies relevant to the original query
        
    Returns:
        average_precision: float corresponding to the AP statistic for this ranking and
        this set of relevant docuemnts.
    '''
    rel_rank = sorted([ranking_in.index(r)+1 for r in relevant])
    return np.mean([(i+1)*1./(r) for i, r in enumerate(rel_rank)])


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
