# Author: Marion Bartl
# Date: 14-7-21
import argparse
import json
import random
from os.path import exists

import wikipedia


def random_page():
    """Adapted from solution by user Banana
    https://stackoverflow.com/questions/43973361/python-retrieve-text-from-multiple-random-wikipedia-pages"""

    random.seed(42)

    # get one random wikipedia article
    rand = wikipedia.random(1)
    try:
        result = (rand, wikipedia.page(title=rand).content)
    # if there is more than one page that fits the query
    except wikipedia.exceptions.DisambiguationError as e:
        result = random_page()
    # if the page can't be found
    except wikipedia.exceptions.PageError as pe:
        result = random_page()
    return result


def wiki_corpus(filename, n_articles=50):
    """download and save wikipedia corpus of n random articles"""

    if exists(filename):
        with open(filename, 'r') as f:
            corpus = json.load(f)
        len_original_corpus = len(corpus)
    else:
        new_corpus = input("Your filename does not exist. "
                           "Do you want to create a new corpus at the location "+filename+" ? (y/n) ")
        if new_corpus == 'y':
            len_original_corpus = 0
            corpus = dict()
        else:
            return

    i = 0
    while i < n_articles:
        pg = random_page()
        if pg[0] not in corpus.keys():
            print('article', str(i + 1), ': ', pg[0])
            corpus[pg[0]] = pg[1]
            i += 1

    # writing corpus to json file
    with open(filename, "w") as outfile:
        json.dump(corpus, outfile, indent=4)

    print('You added', len(corpus) - len_original_corpus, 'random articles to your wikipedia corpus.')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='JSON file of a corpus you\'d  want to create or extend', required=True)
    parser.add_argument('--n', help='number of random articles you want to retrieve', required=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    if args.n:
        wiki_corpus(args.file, int(args.n))
    else:
        wiki_corpus(args.file)
