from .models import Docs
import os
import Levenshtein
import json
import pickle
import difflib


def read_json(n):
	file = open(n)
	j = json.load(file)
	return j

def read_pickle(n):
	p = pickle.load(open(n, 'rb'))
	return p

def query(q, data):
	return difflib.get_close_matches(q, data, cutoff=0.35)[0]

def find_similar(q):
	sims_m = read_pickle('backend_codeV2/book_sims.pickle')
	title_to_idx = read_pickle('backend_codeV2/book_title_to_index.pickle')
	idx_to_title = read_pickle('backend_codeV2/book_index_to_title.pickle')
	ratings = read_pickle('backend_codeV2/ratings.pickle')

	lower_tti = {}
	titles = []

	for k, v in title_to_idx.items():
		titles.append(k)
		lower_tti[k] = v

	q = query(q, titles)
	print("+=================")
	print(q)
	q_idx = lower_tti[q]

	sims_q = sims_m[q_idx]

	result = []

	for idx in range(len(sims_q)):
		sim = sims_q[idx]
		title = idx_to_title[idx]
		result.append((title, sim, float(ratings[title])))

	return q, sorted(result, key=lambda tup: tup[1], reverse=True)