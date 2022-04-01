# Author: Marion Bartl
# Date: 25-8-21
import argparse
import json
import time

import numpy as np
import pandas as pd
import spacy
from nltk import sent_tokenize
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from utils.dict_utils import check_dictionary, grid_search
from utils.general import is_word, conflict_resolution_3
from utils.wiki_utils import parse_wiki


# import neuralcoref


def edit_distance(str1, str2):
    """source: https://www.codespeedy.com/minimum-edit-distance-in-python/
    returns number of operations needed to turn str1 into str2"""

    a = len(str1)
    b = len(str2)
    string_matrix = [[0 for i in range(b + 1)] for i in range(a + 1)]
    for i in range(a + 1):
        for j in range(b + 1):
            if i == 0:
                string_matrix[i][j] = j  # If first string is empty, insert all characters of second string into first.
            elif j == 0:
                string_matrix[i][j] = i  # If second string is empty, remove all characters of first string.
            elif str1[i - 1] == str2[j - 1]:
                string_matrix[i][j] = string_matrix[i - 1][
                    j - 1]  # If last characters of two strings are same, nothing much to do. Ignore the last two
                # characters and get the count of remaining strings.
            else:
                string_matrix[i][j] = 1 + min(string_matrix[i][j - 1],  # insert operation
                                              string_matrix[i - 1][j],  # remove operation
                                              string_matrix[i - 1][j - 1])  # replace operation
    return string_matrix[a][b]


def word_in_series(word, word_series):
    """function that uses a minimum edit distance of 1 to find whether a word
    (or a similar spelling of it) is contained in a series"""

    # additional heuristic:
    # all the strings that start with same first letter as word
    # same_first_letter = word_series[word_series.str.startswith(word[0])]

    for elem in word_series:
        if edit_distance(word, elem) <= 1:
            return True
    else:
        return False


def combine_nouns_lists(nouns_cao, nouns):
    """
    combine gender-specific noun lists from cao & the one I created
    return combined pd.dataframe
    """

    nouns_cao.columns = ['masculine', 'feminine', 'neutral']
    print(len(nouns), len(nouns_cao))

    df = pd.DataFrame(columns=['masculine', 'feminine', 'neutral', 'cao_daume', 'bartl'])
    # all_df = pd.concat([nouns_cao, nouns], keys=['cao', 'bartl'])

    for index, row in nouns_cao.iterrows():
        # masculine
        cao = True
        if row.masculine not in list(df.masculine):
            masc = row.masculine
        else:
            continue
        if row.feminine not in list(df.feminine):
            fem = row.feminine
        else:
            continue

        neutral = row.neutral

        bartl_m = True if word_in_series(row.masculine, nouns.masculine) else False
        bartl_f = True if word_in_series(row.feminine, nouns.feminine) else False

        bartl = bartl_m or bartl_f

        # print([masc, fem, neutral, cao, bartl])
        df.loc[len(df.index)] = [masc, fem, neutral, cao, bartl]

    for index, row in nouns.iterrows():
        # masculine
        bartl = True
        if word_in_series(row.masculine, df.masculine):
            continue
        else:
            masc = row.masculine
        if word_in_series(row.feminine, df.feminine):
            continue
        else:
            fem = row.feminine

        neutral = None

        cao_m = True if word_in_series(row.masculine, nouns_cao.masculine) else False
        cao_f = True if word_in_series(row.feminine, nouns_cao.feminine) else False

        cao = cao_m or cao_f

        # print([masc, fem, neutral, cao, bartl])
        df.loc[len(df.index)] = [masc, fem, neutral, cao, bartl]

    # sort by masculine nouns
    return df.sort_values('masculine', ignore_index=True)


def get_several_metrics(pred_lists, labels, remove_na=False):
    y = []
    assert len(pred_lists) == len(labels)
    for preds, label in zip(pred_lists, labels):
        if remove_na:
            preds.dropna(inplace=True)
        else:
            preds.fillna('not_found', inplace=True)
        y_true = [label] * len(preds)
        y += zip(y_true, preds)

    le = LabelEncoder()

    y_pred = le.fit_transform([p for t, p in y])
    y_true = le.transform([t for t, p in y])

    print(le.classes_)

    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    label2name = {lab: name for lab, name in enumerate(le.classes_)}
    report = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)

    print('Confusion Matrix')
    cf = np.column_stack((le.classes_, confusion_matrix(y_true, y_pred)))
    cf = np.vstack((np.array([''] + list(le.classes_)), cf))
    cf_df = pd.DataFrame(cf)
    print(cf_df.to_string())
    print()

    return report, label2name


def parse_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', help='testing individual words to improve on MW retrieval', action='store_true')
    group.add_argument('--gold', help='gold standard CSV-file (masc, fem & neutral columns)')
    group.add_argument('--wiki', help='wikipedia corpus file in JSON format')

    parser.add_argument('--out_path', help='path to .CSV outfile for found words from wikipedia', required=True)

    parser.add_argument('--heur', help='whether or not to use heuristics for lexical gender algorithm', required=False,
                        action="store_true")
    parser.add_argument('--n_words', help='no_words of dictionary definition to use', required=False,
                        default=10)
    parser.add_argument('--s_pairs', help='no of gendered seed pairs to find in dictionary definition', required=False,
                        default=5)
    parser.add_argument('--n_defs', help='no of definitions to use from dictionary', required=False,
                        default=5)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.test:  # TEST EVALUATION

        test_words = ['contraceptives']

        for test_word in test_words:
            print(check_dictionary(test_word, 'mw'))

    elif args.wiki:  # FINDING WORDS WITH LEXICAL GENDER IN WIKIPEDIA CORPUS
        st = time.time()

        print('Loading the spacy model ...')
        # Load big spacy model
        nlp = spacy.load("en_core_web_lg")
        # add co-reference model to pipeline
        # neuralcoref.add_to_pipe(nlp)
        print('... done')

        # import wikipedia corpus
        wikicorpus = json.load(open(args.wiki, 'r'))
        print('Number of articles in wikicorpus:', len(wikicorpus))

        no_sents = 0
        rows = []
        already_done = set()

        for i, (title, text) in enumerate(wikicorpus.items()):
            if i % 5 == 0 and i > 1:
                print('{} out of {} articles processed.'.format(i, len(wikicorpus)))
            parsed_text = parse_wiki(text)
            article = sent_tokenize(parsed_text)
            processed_sents = list(nlp.pipe(article))
            no_sents += len(article)

            for doc in processed_sents:
                for token in doc:
                    if token.tag_ == 'NN' or token.tag_ == 'NNS':
                        word = token.text.lower()
                        if not is_word(word) or word in already_done:
                            if not is_word(word):
                                print('not a word:', word)
                            continue
                        else:
                            wn_label, mw_label, dc_label = check_dictionary(word, 'wn'), \
                                                           check_dictionary(word, 'mw'), \
                                                           check_dictionary(word, 'dc')
                            comb_label = conflict_resolution_3(wn_label, mw_label, dc_label)
                            rows.append([word, token.tag_, wn_label, mw_label, dc_label, comb_label])
                            already_done.add(word)

        found_words = pd.DataFrame(rows, columns=['word', 'tag', 'wn_label', 'mw_label', 'dc_label', 'comb_label'])
        print(found_words.to_markdown())
        found_words.to_csv(args.out_path, index=False, na_rep='not_found')
        print('Number of sentences in wikicorpus:', no_sents)
        et = time.time()

        print('This took {0:.2f} minutes'.format((et - st) / 60))

    elif args.gold:  # GOLD STANDARD EVALUATION
        st = time.time()

        colnames_gold = ['masculine', 'feminine', 'neutral']
        gender_labels = ['masc', 'fem', 'neutral']

        # list all the dictionaries
        online_dicts = {'merriam': {'fullname': 'Merriam Webster'},
                        'wordnet': {'fullname': 'Wordnet'},
                        'dictcom': {'fullname': 'Dictionary.com'}}

        # get gold standard labels
        gold = pd.read_csv(args.gold)  # watch out for nan values in neutral list

        # turn gold labels into long format while taking care of unseen values in neutral column
        long = []
        for words, label in zip([gold.masculine, gold.feminine, gold.neutral], gender_labels):
            words.dropna(inplace=True)
            y_true = [label] * len(words)
            long += zip(words, y_true)

        gold_long = pd.DataFrame(long, columns=['word', 'true_label'])

        gold_long.to_csv('data/gendered_nouns_gold_standard_long.csv', index=False)

        # create parameter_grid
        param_grid = {'seed_pairs': [2, 3, 4, 5, 6, 7, 8],
                      'no_defs': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'no_words': [5, 10, 15, 20, 25, 30, 35]}

        param_list, online_dicts = grid_search(gold_long.word, gold_long.true_label, param_grid, online_dicts)

        # add combined label
        # gold_and_preds['comb'] = [conflict_resolution_3(a, b, c) for a, b, c
        #                           in zip([gold_and_preds.merriam, gold_and_preds.wordnet, gold_and_preds.dictcom])]

        et = time.time()
        print('This took {0:.2f} minutes'.format((et - st) / 60))

        for abbrev, info in online_dicts.items():
            print(abbrev, info['best_acc'], info['best_params'])

        param_df = pd.DataFrame(param_list)
        param_df.to_csv(args.out_path, index=False)

        # with open(args.out_path, 'w') as f:
        #     json.dump(online_dicts, f, indent=4)

    else:
        print('You haven\'t given any arguments. I don\'t know what to do.')