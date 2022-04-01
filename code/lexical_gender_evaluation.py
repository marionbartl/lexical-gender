import pandas as pd

from utils.general import get_metrics


def get_gender_group(df, gender):
    group = df[df.true_label == gender]
    print('The label {} has {} members'.format(gender, len(group)))
    return group


def get_overlap(wiki, gold):
    overlap = list(set(gold).intersection(set(wiki)))
    overlap.sort()
    print('Overlap between gold & wiki1000: {} words'.format(len(overlap)))
    print(overlap)
    only_gold = list(set(gold).difference(set(wiki)))
    only_gold.sort()
    print('Only in gold:\n', only_gold)
    only_wiki = list(set(wiki).difference(set(gold)))
    only_wiki.sort()
    print('Only in wiki:\n', only_wiki)


if __name__ == '__main__':

    # both dataframes have a 'word' column and a 'true_label' column
    # wiki_data also has 'tag', 'wn_label', 'mw_label' and 'comb_label'
    wiki_data = pd.read_csv('results/lexical_gender_femmasc_wiki1000_tryout.csv', header=0)

    # only get NN nouns
    # wiki_data = wiki_data[wiki_data['tag'] == 'NN']

    gold_data = pd.read_csv('results/lexical_gender_gold_labelled_long.csv', header=0)

    print('OVERLAP ANALYSIS\n')

    # see overlap between the two
    print('Length of Wiki1000: {} nouns'.format(len(wiki_data)))
    # only get singular nouns
    wiki_singles = wiki_data.loc[wiki_data.tag == 'NN']
    wiki_plurals = wiki_data.loc[wiki_data.tag == 'NNS']

    print('{} singular nouns ({:.2f}%), {} plural nouns ({:.2f}%)'.format(len(wiki_singles),
                                                                          len(wiki_singles) / len(wiki_data) * 100,
                                                                          len(wiki_plurals),
                                                                          len(wiki_plurals) / len(wiki_data) * 100))
    print('Gold corpus: {} nouns'.format(len(gold_data)))
    get_overlap(wiki_singles.word, gold_data.word)

    # divide into gender groups
    # print('WIKI1000')
    # wiki_masc_NN = get_gender_group(wiki_data_singles, 'masc')
    # wiki_fem_NN = get_gender_group(wiki_data_singles, 'fem')
    # wiki_neut_NN = get_gender_group(wiki_data_singles, 'neutral')
    # print('GOLD')
    # gold_masc = get_gender_group(gold_data, 'masc')
    # gold_fem = get_gender_group(gold_data, 'fem')
    # gold_neut = get_gender_group(gold_data, 'neutral')
    #
    # print()
    # print('masc'.upper())
    # get_overlap(wiki_masc_NN.word, gold_masc.word)
    # print('fem'.upper())
    # get_overlap(wiki_fem_NN.word, gold_fem.word)
    # print('neut'.upper())
    # get_overlap(wiki_neut_NN.word, gold_neut.word)

    # print('------------------------------------------------')
    # print('CONFLICT ANALYSIS')
    #
    # # quantitative error analysis
    # # columns wn_label, mw_label, true_label
    #
    # dataframes = {'GOLD': gold_data, 'WIKIPEDIA': wiki_data}
    #
    # for data_label, data in dataframes.items():
    #     print('\n' + data_label.upper())
    #     # print(data.to_markdown())
    #
    #     gendered = ['fem', 'masc']
    #     neutral_true, fem_true, masc_true = 0, 0, 0
    #     gendered_conflicts = 0
    #     gen_neut_conflicts = 0
    #     conflicts = 0
    #
    #     neutral_true2, masc_true2, fem_true2 = 0, 0, 0
    #     gendered_true = 0
    #
    #     for i, row in data.iterrows():
    #         if row.mw_label != row.wn_label:
    #             conflicts += 1
    #             if row.mw_label in gendered and row.wn_label in gendered:
    #                 gendered_conflicts += 1
    #                 # print(row.word)
    #                 if row.true_label == 'neutral':
    #                     neutral_true += 1
    #                 elif row.true_label == 'masc':
    #                     masc_true += 1
    #                 elif row.true_label == 'fem':
    #                     fem_true += 1
    #                 else:
    #                     pass
    #                     # print(row.true_label)
    #             else:
    #                 gen_neut_conflicts += 1
    #                 if row.true_label == 'neutral':
    #                     neutral_true2 += 1
    #                 elif row.true_label in gendered:
    #                     # print(row.word, row.mw_label, row.wn_label)
    #                     gendered_true += 1
    #                     if row.true_label == 'masc':
    #                         masc_true2 += 1
    #                     elif row.true_label == 'fem':
    #                         fem_true2 += 1
    #
    #     print('Conflicts happen in {} out of {} cases'.format(conflicts, len(data)))
    #     print('When a noun has a masc/fem conflict ({} out of {}), it is:'.format(gendered_conflicts, conflicts))
    #     print('{} times feminine\n{} times masculine\n{} times neutral'.format(fem_true, masc_true, neutral_true))
    #
    #     print(
    #         'When a noun has a masc/fem vs. neutral conflict ({} out of {}), the true label is:'.format(
    #             gen_neut_conflicts,
    #             conflicts))
    #     print('{} times gendered ({} masc {} fem)\n{} times neutral'.format(gendered_true, masc_true2, fem_true2,
    #                                                                         neutral_true2))

    print('\n------------------------------------------------')

    print('QUANTITATIVE EVALUATION\n')

    # remove words that cause annotator disagreement due to missing word sense disambiguation
    # ambig_words = ['fellow', 'master', 'ram', 'suitor']
    # wiki1000_data = wiki1000_data[~wiki1000_data['word'].isin(ambig_words)]

    columns = {'merriam_webster': 'mw_label',
               'wordnet': 'wn_label',
               'dictionary_com': 'dc_label',
               'combined': 'comb_label'}

    data = {'GOLD': gold_data, 'WIKIPEDIA': wiki_data}

    for df_label, df in data.items():
        print('\n### ' + df_label.upper() + ' ###')
        for c_label, c in columns.items():
            print('\n', c_label.upper())
            results = get_metrics(df[c], df.true_label)
            s = df[c]
            print('Words that could not be found by dictionary:')
            # get all the words for which it is true that their label wasn't found
            print(df.word[s.isin(['not_found'])].to_string(index=False))

        neut_as_gendered = df[(df['true_label'] == 'neutral') & df['comb_label'].isin(['fem', 'masc'])]
        print('true neutral words classified as gendered')
        print(neut_as_gendered.loc[:, ['word','dc_label', 'mw_label', 'wn_label', 'comb_label', 'true_label']].sort_values('comb_label'))
        print()
        gendered_as_neut = df[(df['true_label'].isin(['fem', 'masc']) & (df['comb_label'] == 'neutral'))]
        print('true gendered words classified as neutral')
        print(gendered_as_neut.loc[:, ['word','dc_label', 'mw_label', 'wn_label', 'comb_label', 'true_label']].sort_values('true_label'))

