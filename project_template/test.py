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
	sims_m = read_json('backend_codeV1/book_sims.json')
	title_to_idx = read_pickle('backend_codeV1/book_title_to_index.pickle')
	idx_to_title = read_pickle('backend_codeV1/book_index_to_title.pickle')

	lower_tti = {}
	titles = []

	for k, v in title_to_idx.items():
		k = k.lower()
		titles.append(k)
		lower_tti[k] = v

	q = query(q, titles)
	q_idx = lower_tti[q]

	sims_q = sims_m[q_idx]

	result = []

	for idx in range(len(sims_q)):
		if (idx==71 or idx==72 or idx==74 or idx==77 or idx==91 or idx==92
			or idx==120 or idx==121 or idx==122 or idx==123 or idx==124
			or idx==163):
			continue
		else:
			sim = sims_q[idx]
			title = idx_to_title[idx]
			result.append((sim, title))

	return sorted(result, key=lambda tup: tup[0], reverse=True)