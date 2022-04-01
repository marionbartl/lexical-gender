# Author: Marion Bartl

import re
import time
from collections import Counter

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid

from .general import conflict_resolution_3
from .morphosyntactic_utils import suffix_heuristics


def gendered_word_count(definition_words, seed_sets):
    female_set = seed_sets['female']
    male_set = seed_sets['male']
    # count all words in the noun description
    def_words = Counter(definition_words)

    # record how often female/male words are mentioned in noun description
    female_found = {f: def_words[f] for f in female_set}
    male_found = {m: def_words[m] for m in male_set}

    # print(female_found)
    # print(male_found)

    # return dict noun gender based on which gender words were mentioned more often
    if sum(female_found.values()) > sum(male_found.values()):
        label = 'fem'
    elif sum(female_found.values()) < sum(male_found.values()):
        label = 'masc'
    elif sum(female_found.values()) == sum(male_found.values()):
        label = 'neutral'
    else:
        return

    return label


def check_wordnet(word, seed_sets, no_words=10, no_defs=1):
    # deal with compounds
    if len(word.split()) > 1:
        word = '_'.join(word.split())

    # get all synsets
    result = wn.synsets(word, pos=wn.NOUN)
    # print('no_of defs:', len(result))

    def_words = []
    if not result:
        # print('Not in wordnet: ', word)
        # check if there is a different spelling without non-word character that can be found.
        non_word_char = re.search(r'[\W_]', word)
        if non_word_char:
            return check_wordnet(word.replace(non_word_char.group(), ''), seed_sets)
        else:
            return

    return [item.definition() for item in result]

    # for item in result[:no_defs]:
    #     # print(item.definition())
    #     def_words += word_tokenize(item.definition())[:no_words]
    #
    # return gendered_word_count(def_words, seed_sets)


def check_merriam_webster(word, seed_sets, no_words=10, no_defs=1):
    female_set = seed_sets['female']
    male_set = seed_sets['male']

    seed_url = "https://www.merriam-webster.com/dictionary/{}"

    # deal with compounds
    if len(word.split()) > 1:
        word = '%20'.join(word.split())

    # time.sleep(2)
    # print(word)

    formatted_url = seed_url.format(word)
    html_text = requests.get(formatted_url).text
    soup = BeautifulSoup(html_text, "html.parser")
    # print(f"Now Scraping - {formatted_url}")

    # find all spans with class "sb-n" if n unknown
    def_class = re.compile(r'^sb-\d{1,2}?$')
    definitions = soup.find_all('span', {'class': def_class})
    # see if we land on a redirection page
    outside_reference = soup.find_all('a', {'class': 'cxt'})

    # if no definitions can be found, check if word leads to re-direction page
    if not definitions:
        misspelled = soup.find_all('h1', {'class': 'mispelled-word'})
        if misspelled:
            # look for non-word character; if found, remove and try again with the result
            # this is for words like grand-father, which can only be found without the dash
            non_word_char = re.search(r'[\W_]', word)
            if non_word_char:
                return check_merriam_webster(word.replace(non_word_char.group(), ''), seed_sets)
            else:
                return
        elif outside_reference:
            referred_word = outside_reference[0].get_text()
            print('we were referred from {0} to {1}'.format(word, referred_word))
            return check_merriam_webster(referred_word.lower(), seed_sets)
        else:
            # if not misspelled & not found, I used to raise an Exception. Now printing to keep the flow
            print(word + ' is not misspelled but has no definition. Please double check.')
    else:
        if outside_reference:
            # check the ratio of outside references to definitions. If there are more outside refs than defs,
            # use the outside ref, else the definitions
            def_ref_ratio = len(outside_reference) / (len(outside_reference) + len(definitions))
            # print('word: {0}({2}), ref: {1}({3}), ratio: {4:.2f}'.format(word, outside_reference[0].get_text(),
            #                                                          len(definitions), len(outside_reference),
            #                                                          def_ref_ratio))

            # if outside references exist, make sure that they are less than the definitions first.
            # if there are more references than definitions, we assume that we are really looking for the
            # referred word and not the original one.
            if def_ref_ratio > 0.5:
                referred_word = outside_reference[0].get_text()
                print('we were referred from {0} to {1}'.format(word, referred_word))
                return check_merriam_webster(referred_word.lower(), seed_sets)

        # def_words = set()  # set alternative
        def_words = []
        # print('No. of definiton texts:', len(definitions))
        for d in definitions[:no_defs]:
            # span with class 'dtText' holds the actual description
            def_texts = [elem.get_text() for elem in d.find_all('span', {'class': 'dtText'}) if elem]
            # if def_texts:
            #     for text in def_texts:
            #         # def_words = def_words.union(set(text.split()))  # set alternative
            #         # the first character is always ':', so I'm taking the no_words + 1
            #         def_words += word_tokenize(text)[:no_words + 1]

        return def_texts

        #### VERSION WITH SETS #####
        # problem: some male words occur in female noun descriptions and vice versa

        # check if the descriptions contained male/female words
        # female_found = def_words.intersection(female)
        # male_found = def_words.intersection(male)
        # print(male_found)
        # print(female_found)

        # return dictionary gender label ...
        # if not female_found and not male_found:
        #     return 'neutral'
        # elif female_found and not male_found:
        #     return 'fem'
        # elif male_found and not female_found:
        #     return 'masc'
        # else:  # ... or the word itself if the dict gender label was inconclusive
        #     return word


def check_dict_com(word, seed_sets, no_words=10, no_defs=1):
    female_set = seed_sets['female']
    male_set = seed_sets['male']

    seed_url = "https://www.dictionary.com/browse/{}"

    # deal with compounds

    if len(word.split()) > 1:
        word = '-'.join(word.split())

    formatted_url = seed_url.format(word)
    html_text = requests.get(formatted_url).text
    soup = BeautifulSoup(html_text, "html.parser")

    # # find all divs
    definitions = soup.find_all('div')
    if definitions:
        def_texts = []
        for d in definitions:
            if 'class' in d.attrs.keys():
                # alternative for spans
                # if d['class'] == "one-click-content css-nnyc96 e1q3nk1v1".split():
                if d['class'] == "css-10ul8x e1q3nk1v2".split():
                    def_texts.append(d.text)
        return def_texts
        # def_words = []
        # for text in def_texts[:no_defs]:
        #     def_words += word_tokenize(text)[:no_words + 1]
        #
        # return gendered_word_count(def_words, seed_sets)
    else:
        print(word + ': word misspelled or not in dictionary')
        exit()


def check_dictionary(word, dict_abbrev, grid=False, heuristics=True, seed_pairs=5, no_words=20, no_defs=4):
    # check if word exists in the first place
    if not word or isinstance(word, float):
        return

    definitive_female = ['woman', 'female', 'wife', 'daughter', 'mother', 'girl', 'sister', 'aunt']
    definitive_male = ['man', 'male', 'husband', 'son', 'father', 'boy', 'brother', 'uncle']

    female_plurals = [word + 's' for word in definitive_female[1:]] + ['women']
    male_plurals = [word + 's' for word in definitive_male[1:]] + ['men']

    female = set(definitive_female + female_plurals)
    male = set(definitive_male + male_plurals)

    seed_sets = {'female': set(definitive_female[:seed_pairs]),
                 'male': set(definitive_male[:seed_pairs])}

    if word in female:
        # print('Word contained in female set')
        return 'fem'
    if word in male:
        # print('Word contained in male set')
        return 'masc'

    if heuristics:
        heur_label = suffix_heuristics(word)
        if heur_label:
            # print('Suffix heuristics were used for {}. label: {}'.format(word, heur_label))
            return heur_label

    if dict_abbrev in ['merriam', 'mw', 'MW']:
        if grid:
            return check_merriam_webster(word, seed_sets, no_words, no_defs)
        else:
            def_list = check_merriam_webster(word, seed_sets, no_words, no_defs)
    elif dict_abbrev in ['wn', 'WN', 'wordnet']:
        if grid:
            return check_wordnet(word, seed_sets, no_words, no_defs)
        else:
            def_list = check_wordnet(word, seed_sets, no_words, no_defs)
    elif dict_abbrev in ['dictcom', 'DC', 'dc']:
        if grid:
            return check_dict_com(word, seed_sets, no_words, no_defs)
        else:
            def_list = check_dict_com(word, seed_sets, no_words, no_defs)
    else:
        raise ValueError('The dictionary abbreviation', dict_abbrev, 'could not be found!')

    if not grid:
        if not def_list:
            return 'not_found'
        else:
            def_words = []
            for text in def_list[:no_defs]:
                def_words += word_tokenize(text)[:no_words + 1]

            # get label based on word count of gendered words
            label = gendered_word_count(def_words, seed_sets)
            if label:
                return label
            else:
                return 'not_found'


def dict_labels(word_lists, cols, heuristics=True, no_words=10, seed_pairs=5):
    label_dfs = []
    for i, wl in enumerate(word_lists):
        print('checking list {0} of {1}: {2}'.format(i + 1, len(word_lists), cols[i]))
        wn_labels = [check_dictionary(word, dict_abbrev='wn', heuristics=heuristics,
                                      no_words=no_words, seed_pairs=seed_pairs) for word in wl]
        mw_labels = [check_dictionary(word, dict_abbrev='mw', heuristics=heuristics,
                                      no_words=no_words, seed_pairs=seed_pairs) for word in wl]
        dc_labels = [check_dictionary(word, dict_abbrev='dc', heuristics=heuristics,
                                      no_words=no_words, seed_pairs=seed_pairs) for word in wl]
        # for w, l in zip(wl, dc_labels):
        #     print(w, l)

        # combining the labels combined_labels = [conflict_resolution_2(wn_label, mw_label) for wn_label, mw_label in
        # zip(wn_labels, mw_labels)]
        combined_labels = [conflict_resolution_3(wn_label, mw_label, dc_label) for wn_label, mw_label, dc_label
                           in zip(wn_labels, mw_labels, dc_labels)]

        # for w, wn, mw in zip(wl, wn_labels, mw_labels):
        #     c = conflict_resolution_2(wn, mw)
        #     print(w, wn, mw, ':', c)

        label_df = pd.DataFrame(list(zip(wl, mw_labels, wn_labels, dc_labels, combined_labels)),
                                columns=[cols[i],
                                         cols[i] + '_merriam',
                                         cols[i] + '_wordnet',
                                         cols[i] + '_dictcom',
                                         cols[i] + '_comb'])

        label_dfs.append(label_df)
        print('done!')

    words_plus_labels = pd.concat(label_dfs, axis=1)
    # print(words_plus_labels.to_markdown())
    return words_plus_labels


def grid_search(X_words, y_true, param_grid_dict, online_dicts):
    # create parameter_grid
    param_grid = ParameterGrid(param_grid_dict)

    # seed sets
    definitive_female = ['woman', 'female', 'wife', 'daughter', 'mother', 'girl', 'sister', 'aunt']
    definitive_male = ['man', 'male', 'husband', 'son', 'father', 'boy', 'brother', 'uncle']

    # go through all dicts
    for abbrev, info in online_dicts.items():
        st2 = time.time()
        print(info['fullname'].upper())  # print name of dict to provide orientation to user

        # instances in X_defs can either be a label, a list of definitions, or 'not found'
        X_defs = [check_dictionary(word, dict_abbrev=abbrev, grid=True) for word in X_words]
        online_dicts[abbrev]['X_defs'] = X_defs

        online_dicts[abbrev]['best_acc'] = 0
        online_dicts[abbrev]['best_params'] = 0

    online_dicts['comb'] = dict()
    online_dicts['comb']['best_acc'] = 0
    online_dicts['comb']['best_params'] = 0

    # pre-prediction settings
    param_list = []  # list that captures all the parameters
    pg = iter(param_grid)
    for i in range(len(param_grid)):
        grid = next(pg)
        answers_df = pd.DataFrame()
        for abbrev, info in online_dicts.items():
            if abbrev == 'comb':
                continue
            else:
                y_pred = []  # here we save all three labels for one grid run
                for inst in info['X_defs']:
                    if inst:
                        if isinstance(inst, list):
                            def_words = []
                            for text in inst[:grid['no_defs']]:
                                def_words += word_tokenize(text)[:grid['no_words'] + 1]

                            seed_sets = {'female': set(definitive_female[:grid['seed_pairs']]),
                                         'male': set(definitive_male[:grid['seed_pairs']])}
                            # get label based on word count of gendered words
                            label = gendered_word_count(def_words, seed_sets)
                            if label:
                                y_pred.append(label)
                            else:
                                y_pred.append('not_found')
                        else:  # case in which the label has already been applied through heuristic
                            y_pred.append(inst)
                    else:
                        y_pred.append('not_found')

                assert len(y_pred) == len(y_true)
                answers_df[abbrev] = y_pred

        answers_df['comb'] = [conflict_resolution_3(row.merriam, row.wordnet, row.dictcom) for i, row in
                              answers_df.iterrows()]

        for name, col in answers_df.iteritems():
            assert len(col) == len(y_true)
            acc = accuracy_score(y_true, col)
            grid.update({'accuracy_' + str(name): round(acc, 2)})
            if acc > online_dicts[str(name)]['best_acc']:
                online_dicts[str(name)]['best_acc'] = round(acc, 2)
                online_dicts[str(name)]['best_params'] = grid

        param_list.append(grid)

    # print(best_acc, best_params)
    et2 = time.time()
    print('Grid search took {0:.2f} minutes'.format((et2 - st2) / 60))

    return param_list, online_dicts
