# Author: Marion Bartl

import numpy as np
import pandas as pd

from utils.dict_utils import check_dictionary
from utils.general import conflict_resolution_3


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
    """function that uses minimum edit distance to find whether a word is contained in a series"""
    # all the strings that start with same first letter as masc
    # same_first_letter = word_series[word_series.str.startswith(word[0])]
    for elem in word_series:
        if edit_distance(word, elem) <= 1:
            return True
    else:
        return False


def fleiss_kappa(M):
    """
    source: https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """

    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)

    return kappa


def cohen_kappa(ann1, ann2):
    # https://towardsdatascience.com/inter-annotator-agreement-2f46c6d37bf3
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)


if __name__ == '__main__':
    # words are all in one column
    # random_data = pd.read_csv('results/lexical_gender_femmasc_wiki150.csv', header=0)

    # gold_data = pd.read_csv('results/gendered_nouns_gold_standard_long_labelled.csv', header=0)
    #
    # for mw, wn in zip(gold_data.mw_label, gold_data.wn_label):
    #     label = conflict_resolution_specific(mw, wn)

    # ------------------------------------------------

    # # CODE TO MAKE LONG DATAFRAME OUT OF GOLD STANDARD FILE
    gold_data = pd.read_csv('data/gendered_nouns_gold_standard.csv', header=0)

    male = list(zip(gold_data.masculine, ['masc']*len(gold_data.masculine)))
    female = list(zip(gold_data.feminine, ['fem']*len(gold_data.masculine)))

    neut = gold_data.neutral.dropna()
    neutral = list(zip(neut, ['neutral']*len(neut)))

    reordered_gold = pd.DataFrame(male+female+neutral, columns=['word', 'true_label'])

    print(reordered_gold.to_markdown())
    reordered_gold.to_csv('data/gendered_nouns_gold_standard_long.csv', index=False)

    # ------------------------------------------------

    # ADDING COMBINED LABEL COLUMN TO PREDICTED DF

    # data = pd.read_csv('results/lexical_gender_all_wiki150.csv')
    #
    # comb_labels = []
    # for idx, row in data.iterrows():
    #     c_label = conflict_resolution_2(row.mw_label, row.wn_label)
    #     comb_labels.append(c_label)
    #
    # data.insert(loc=4, column='comb_label', value=comb_labels)
    # data.to_csv(path_or_buf='results/lexical_gender_all_wiki150.csv', index=False)

    # ------------------------------------------------

    # SPACY TRYOUTS

    # nlp = spacy.load('en_core_web_lg')
    # print('model loaded')
    # neuralcoref.add_to_pipe(nlp)
    # print('neuralcoref added')
    # doc = nlp('The dog Biscuit can be seen on his daily walks.')
    # print(doc.ents)
    # for token in doc:
    #     if token.text in [token.text for token in doc.ents]:
    #         ne = 'NE'
    #     else:
    #         ne = ''
    #     print(token.text, token.tag_, token.dep_, ne)
    #
    # for cluster in doc._.coref_clusters:
    #     print(cluster.main.text)
    #     print(cluster.mentions)


    # ------------------------------------------------

    # CODE TO NOT DO GENDER LABELLING AGAIN AND AGAIN

    # new_wiki_results = pd.read_csv('results/lexical_gender_femmasc_wiki150_new.csv')
    # new_wiki_results.sort_values('word', inplace=True)
    # old_wiki_results = pd.read_csv('results/lexical_gender_femmasc_wiki150.csv')
    # old_wiki_results.sort_values('word', inplace=True)
    #
    # assert len(new_wiki_results) == len(old_wiki_results)
    #
    # manual_labels = list(old_wiki_results.true_label)
    # new_wiki_results['true_label'] = manual_labels
    #
    # new_wiki_results.to_csv(path_or_buf='results/lexical_gender_femmasc_wiki150.csv', index=False)

    # ------------------------------------------------
    # POST-HOC LABELING OF WIKIPEDIA DATA

    # wiki = pd.read_csv('results/lexical_gender_femmasc_wiki150.csv', header=0)
    #
    # # mw_labels = [check_dictionary(word, dict_abbrev='mw') for word in wiki.word]
    #
    # mw_labels = []
    # wn_labels = []
    # c_labels = []
    # for i, word in enumerate(wiki.word):
    #     # print(word)
    #     lm = check_dictionary(word, dict_abbrev='mw')
    #     lw = check_dictionary(word, dict_abbrev='wn')
    #     # print(l, wiki.true_label[i])
    #     # print()
    #     mw_labels.append(lm)
    #     wn_labels.append(lw)
    #     c_labels.append(conflict_resolution_2(lm, lw))
    #
    # wiki['wn_label'] = wn_labels
    # wiki['mw_label'] = mw_labels
    # wiki['comb_label'] = c_labels
    #
    # wiki.to_csv(path_or_buf='results/lexical_gender_femmasc_wiki150.csv', index=False)
    #
    # # for word, label, t in zip(wiki.word, mw_labels, wiki.true_label):
    # #     print(word, label, t)
    #
    # # check_df = pd.DataFrame(zip(wiki.word, mw_labels, wiki.true_label), columns=['word', 'mw', 'true'])
    #
    # problems = wiki[wiki.mw_label != wiki.true_label]
    # print(problems[['word', 'mw_label', 'true_label']].to_markdown())
    #
    # print(get_metrics(pd.Series(mw_labels), wiki.true_label))

    # ------------------------------------------------
    # COMPARISON OF CAO's and our

    # gold_cao = pd.read_csv('data/sem.csv')
    # gold_bartl = pd.read_csv('data/gendered_nouns_gold_standard.csv')
    # print(gold_bartl.to_markdown())
    #
    # colnames = ['masculine', 'feminine', 'neutral']
    #
    # gold_cao.rename(columns={'Masculine': 'masculine',
    #                          'Feminine': 'feminine',
    #                          'Neutral': 'neutral'}, inplace=True)
    #
    # # remove outdated/offensive words
    # # gold_cao.loc[gold_cao.neutral == 'REMOVE', 'neutral'] = np.nan
    # gold_cao = gold_cao[gold_cao.neutral != 'REMOVE']
    #
    # for i, row in gold_cao.iterrows():
    #     new_row = [np.nan]
    #     if row.masculine not in list(gold_bartl.masculine):
    #         new_row.append(row.masculine)
    #     if row.feminine not in list(gold_bartl.feminine):
    #         new_row.append(row.feminine)
    #     if row.neutral not in list(gold_bartl.neutral):
    #         new_row.append(row.neutral)
    #
    #     if len(new_row) == 4:
    #         if new_row[1] != new_row[3]:  # masc != neutral
    #             gold_bartl.loc[len(gold_bartl)] = new_row
    #
    # for gender in colnames:
    #     c = set(gold_cao[gender].dropna())
    #     b = set(gold_bartl[gender].dropna())
    #     intsec = c.intersection(b)
    #     diff_cao = c.difference(b)
    #     diff_ba = b.difference(c)
    #     print(gender)
    #     print('# cao: {0}, # bartl: {1}'.format(len(c), len(b)))
    #     print('# intersection: {0}, # only in cao: {1}, # only in bartl: {2}'.format(len(intsec), len(diff_cao),
    #                                                                                  len(diff_ba)))
    # # print(intsec, '\n', diff)
    #
    # gold_cao.to_csv('gendered_nouns_gold_standard_cao_altered.csv', index=False)
    #
    # print(gold_bartl.to_markdown())
    # gold_bartl.to_csv('data/gendered_nouns_gold_standard.csv', index=False, index_label=False)

    # ------------------------------------------------

    # ORDERING THE GOLD STANDARD

    # gold = pd.read_csv('data/gendered_nouns_gold_standard.csv')
    # gold.sort_values(['category', 'masculine'],
    #                ascending=[True, True], inplace=True)
    # gold.to_csv('data/gendered_nouns_gold_standard.csv', index=False)
    # print(gold.to_markdown())
    #
    # print(gold.info())

    # ------------------------------------------------

    # POST-HOC LABELING OF WIKIPEDIA DATA after conf_res_3 was changed

    # wiki = pd.read_csv('results/gendered_nouns_wiki1000_sample_majority.csv', header=0)
    # print(wiki.comb_label.value_counts())
    #
    # new_comb_labels = []?
    #
    # for index, row in wiki.iterrows():
    #     new_comb_labels.append(conflict_resolution_3(row.mw_label, row.dc_label, row.wn_label))
    #
    # wiki['comb_label'] = new_comb_labels
    #
    # print(wiki.comb_label.value_counts())
    # wiki.to_csv('results/gendered_nouns_wiki1000_sample_majority.csv', index=False)

