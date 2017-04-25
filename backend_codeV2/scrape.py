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
	api_key = ''
	secret  = ''
	gc = client.GoodreadsClient(api_key, secret)
	index = {}

	for i in range(1000, 2000):
		try:
			bk = gc.book(i)
			print('On BookID: ' + str(i))
			if bk.title in index or not bk.description:
				continue
			index[str(bk.title)] = [str(bk.description), str(bk.average_rating)]
			time.sleep(1)
		except:
			continue

	create_pickle(index, 'data2.pickle')

main()

