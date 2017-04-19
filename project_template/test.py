from .models import Docs
import os
import Levenshtein
import json
import pickle


def read_json(n):
	file = open(n)
	j = json.load(file)
	return j

def read_pickle(n):
	p = pickle.load(open(n, 'rb'))
	return p

def query(q):
	return q.lower()

def find_similar(q):
	sims_m = read_json('backend_code/book_sims.json')
	title_to_idx = read_pickle('backend_code/book_title_to_index.pickle')
	idx_to_title = read_pickle('backend_code/book_index_to_title.pickle')

	lower_tti = {}

	for k, v in title_to_idx.items():
		k = k.lower()
		lower_tti[k] = v

	q = query(q)
	q_idx = lower_tti[q]

	sims_q = sims_m[q_idx]

	result = []

	for idx in range(len(sims_q)):
		if (idx==145):
			continue
		else:
			sim = sims_q[idx]
			title = idx_to_title[idx]
			result.append((sim, title))

	return sorted(result, key=lambda tup: tup[0], reverse=True)