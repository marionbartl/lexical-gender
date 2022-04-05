import pandas as pd

from utils.general import get_metrics


def get_gender_group(df, gender):
    group = df[df.true_label == gender]
    print('The label {} has {} members'.format(gender, len(group)))
    return group


def get_overlap(wiki, gold):
    overlap = list(set(gold).intersection(set(wiki)))
    overlap.sort()
    print('Overlap between gold & wiki1000-sample ({} words):'.format(len(overlap)))
    print(overlap, '\n')
    only_gold = list(set(gold).difference(set(wiki)))
    only_gold.sort()
    print('Only in gold ({} words):'.format(len(only_gold)))
    print(only_gold, '\n')
    only_wiki = list(set(wiki).difference(set(gold)))
    only_wiki.sort()
    print('Only in wiki1000-sample ({} words):'.format(len(only_wiki)))
    print(only_wiki, '\n')


if __name__ == '__main__':

    # both small datasets have a 'word' column and a 'true_label' column
    # wiki_data also has 'tag', 'wn_label', 'mw_label' and 'comb_label'
    wiki_sample = pd.read_csv('results/gendered_nouns_wiki1000_sample_majority.csv', header=0)
    gold_data = pd.read_csv('results/gendered_nouns_gold_standard_long_labelled.csv', header=0)
    # wiki1000 has the columns ['word', 'tag', 'wn_label', 'mw_label', 'dc_label', 'comb_label'] but not 'true_label'
    wiki1000 = pd.read_csv('results/gendered_nouns_wiki1000.csv', header=0)

    # only get NN nouns
    # wiki_data = wiki_data[wiki_data['tag'] == 'NN']

    print('1. Category Count\n'.upper())

    print('Gold Standard')
    gold_tally = gold_data.groupby(['true_label']).agg({'true_label': 'count'})
    print(gold_tally, '\n')

    print('Wiki Corpora')
    wikiS_tally = wiki_sample.groupby(['tag', 'true_label']).agg({'tag': 'count'})
    wiki1000_tally = wiki1000.groupby(['tag', 'comb_label']).agg({'tag': 'count'})

    wiki_tally = pd.concat([wikiS_tally, wiki1000_tally], axis=1)
    wiki_tally.columns = ['wiki-sample', 'wiki_1000']

    print(wiki_tally)

    print('\n------------------------------------------------')

    print('2. QUANTITATIVE & QUALITATIVE EVALUATION\n')

    columns = {'merriam_webster': 'mw_label',
               'wordnet': 'wn_label',
               'dictionary_com': 'dc_label',
               'combined': 'comb_label'}

    data = {'GOLD STANDARD': gold_data, 'WIKIPEDIA': wiki_sample}

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
        print('True neutral words classified as gendered')
        print(neut_as_gendered.loc[:,
              ['word', 'dc_label', 'mw_label', 'wn_label', 'comb_label', 'true_label']].sort_values('comb_label'))
        print()
        gendered_as_neut = df[(df['true_label'].isin(['fem', 'masc']) & (df['comb_label'] == 'neutral'))]
        print('True gendered words classified as neutral')
        print(gendered_as_neut.loc[:,
              ['word', 'dc_label', 'mw_label', 'wn_label', 'comb_label', 'true_label']].sort_values('true_label'))

    print('\n------------------------------------------------')

    print('3. OVERLAP ANALYSIS\n')

    # see overlap between wiki and gold data
    print('Length of Wiki1000-sample: {} nouns'.format(len(wiki_sample)))
    # split up singular and plural nouns
    wiki_singles = wiki_sample.loc[wiki_sample.tag == 'NN']
    wiki_plurals = wiki_sample.loc[wiki_sample.tag == 'NNS']

    print('{} singular nouns ({:.2f}%), {} plural nouns ({:.2f}%)'.format(len(wiki_singles),
                                                                          len(wiki_singles) / len(wiki_sample) * 100,
                                                                          len(wiki_plurals),
                                                                          len(wiki_plurals) / len(wiki_sample) * 100))
    print('Gold corpus: {} nouns'.format(len(gold_data)))
    print()
    get_overlap(wiki_singles.word, gold_data.word)

    print('\n------------------------------------------------')