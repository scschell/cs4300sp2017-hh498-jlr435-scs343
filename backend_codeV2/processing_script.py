""" Contains back-end scripts needed for processing pickles.
Author @ Jorge Rocha <jlr435@cornell.edu>
Inspired from a script by @ Sayge Schell <scs343@cornell.edu>
"""

import cPickle as pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import numpy as np

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

def generate_dicts(d):
	descripts = {}
	stars     = {}
	book_to_idx = {}
	idx_to_book = {}
	i = 0
	for key in d.keys():
		descripts[key] = d[key][0]
		stars[key]     = d[key][1]
		book_to_idx[key] = i
		idx_to_book[i] = key
		i += 1
	return descripts, stars, book_to_idx, idx_to_book

def clean_trash(tokens):
	result = []
	for token in tokens:
		if (token == 'b' or token == 'br' or token == 'i'
		or token == 'em' or token == 'p'):
			continue
		else:
			result.append(token)
	return result

def clean_descripts(descripts):
	tokenizer = RegexpTokenizer(r'\w+')
	clean_descripts = {}
	for key in descripts.keys():
		token_list = tokenizer.tokenize(descripts[key])
		result = clean_trash(token_list)
		clean_descripts[key] = result
	return clean_descripts

"""Build tf-idf vectors of script data.
    Input: list of dictionaries containing book data
    Output: index to vocab list, book by vocab vector array
"""
def build_vectors(data):
    n_feats = 1800
    book_by_vocab = np.empty([len(data), n_feats])
    tfidf_vec = TfidfVectorizer(input=data, min_df=10, max_df=.8, max_features=n_feats, stop_words=None, norm='l2')
    #extract scripts
    scripts = [data[key] for key in data.keys()]
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

"""Builds book descriptions similarity vector
    Input:
        book_descripts: matrix of books and their tokenized descriptions
        book_index_to_title: dict mapping book indices to titles
        book_title_to_index: dict mapping book titles to indices
    Ouput: book similarity vector"""
def build_similarities(book_descripts, data, book_index_to_title, book_title_to_index):
    book_sims = np.empty([len(data), len(data)], dtype = np.float32)
    #For each pair of items
    for i in range(0, len(book_descripts)):
        for j in range(0, len(book_descripts)):
            if i not in book_index_to_title or j not in book_index_to_title:
                book_sims[i][j] = -1
            #If it is the diagonal, set to 0
            elif i == j:
                book_sims[i][j] = 0
            #Otherwise, find the similarity
            else :
                book_sims[i][j] = get_sim(book_index_to_title[i], book_index_to_title[j], book_title_to_index, book_descripts)
    return book_sims

def main():
	print("STARTING PROCESSING SCRIPT")
	print("LOADING DATA")
	data1 = load_pickle('data1.pickle')
	data2 = load_pickle('data2.pickle')
	data3 = load_pickle('data3.pickle')
	print("LOADED DATA")

	print("MERGING DATA")
	data  = merge_dicts(data1, data2, data3)
	print("MERGED DATA")
	descriptions, ratings, book_to_idx, idx_to_book = generate_dicts(data)
	tokenized_descripts = clean_descripts(descriptions)

	print("BUILDING VECTORS")
	_, book_by_vocab = build_vectors(descriptions)
	print("BUILT VECTORS")

	print("MAKING BOOK SIMS")
	book_sims = build_similarities(book_by_vocab, data, idx_to_book, book_to_idx)
	print("BUILT BOOK SIMS")

	print("MAKING PICKLES")
	create_pickle(book_sims, "book_sims.pickle")
	create_pickle(ratings, "ratings.pickle")
	create_pickle(descriptions, "descriptions.pickle")
	create_pickle(book_to_idx, "book_title_to_index.pickle")
	create_pickle(idx_to_book, "book_index_to_title.pickle")

	print("DONE")

main()
