from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import re
import os
import cPickle as pickle
import time
from goodreads import client

def create_pickle(item, name):
    with open(name, 'wb') as f:
        pickle.dump(item, f)

def main():
	api_key = 'ZdhASvKYwV3oFbdGxHVLA'
	secret  = '69QbHfonpppJBJ2oIavrsDzbAefWcGiBb0R5HrvXz8'
	gc = client.GoodreadsClient(api_key, secret)
	index = {}
	author_index = {}

	print("STARTING SCRAPING")
	for i in range(8000, 10000):
		try:
			bk = gc.book(i)
			print('On BookID: ' + str(i))
			if bk.title in index or not bk.authors or not bk.description:
				continue
			index[str(bk.title)] = [str(bk.description), str(bk.average_rating)]
			author_index[str(bk.title)] = [a.name for a in bk.authors]
			time.sleep(1)
		except:
			continue


	print("MAKING PICKLES")
	create_pickle(index, 'data6.pickle')
	create_pickle(author_index, 'data6_authors.pickle')
	print("DONE")

main()

