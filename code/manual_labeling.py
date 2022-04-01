import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from utils.general import get_metrics


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='testing individual words to improve on MW retrieval', required=True)
    parser.add_argument('--comp', help='data against which the new to label data will be compared', required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    data = pd.read_csv(args.data, header=0)
    if args.comp:
        comparison = pd.read_csv(args.comp, header=0)
        print('The comparison file was read in!')

    # labels: word, tag, wn_label, mw_label

    instructions = """You will be shown a bunch of words. 
    Please label them with their lexical gender. 
    The lexical genders available to you are male (m), female (f) and neutral (n).\n """
    print(instructions)

    continued_labelling = True if 'true_label' in data.columns else False
    if not continued_labelling:
        data['true_label'] = ""
        print('No words have been labelled yet.')
    else:
        print('You seem to have already started to label the words.')


    if args.comp:
        from_comp = 0
        for idx, row in data.iterrows():
            if isinstance(row.true_label, str):
                continue
            elif row.word in list(comparison.word):  # and not row['true_label']:
                # print(row.word, 'is already contained in the comparison file.')
                true_label = comparison.loc[comparison['word'] == row.word].true_label.item()  # item turns it into str
                # print('its label is', true_label)
                data.loc[idx, 'true_label'] = true_label
                from_comp += 1
        print(from_comp, 'labels were pulled from comparison file.')

    # sort data so that nans are last and we can count easier how many we have done
    data = data.sort_values(['true_label', 'word']).reset_index(drop=True)
    print(len(data)-data['true_label'].isnull().sum(), 'out of', len(data), 'labels are already there.')

    cont = True if input('Do you want to start labelling? (y/n) ') == "y" else False
    if cont:
        for idx, row in data.iterrows():
            if isinstance(row.true_label, str):
                data.loc[idx, 'true_label'] = row.true_label
                continue
            if idx % 10 == 0 and idx > 0:
                print('You got {} out of {}.'.format(idx, len(data)))
                cont = True if input('Do you want to continue? (y/n) ') == "y" else False
                if not cont:
                    break
            while True:
                true_l = input('word: {}   gender label: '.format(row.word.upper(), row.mw_label))
                if true_l in {'m', 'masc', 'M'}:
                    data.loc[idx, 'true_label'] = 'masc'
                    break
                elif true_l in {'f', 'fem', 'F'}:
                    data.loc[idx, 'true_label'] = 'fem'
                    break
                elif true_l in {'n', 'N', 'neutral', 'neut'}:
                    data.loc[idx, 'true_label'] = 'neutral'
                    break
                else:
                    print('try again.')

    # #if len(data.true_label) < len(data):
    #     true_labels = true_labels + [""] * (len(data) - len(true_labels))
    #     data['true_label'] = true_labels

    # if len(data.true_label) == len(data):
    if data['true_label'].isnull().sum() == 0:
        pass
        print('The data have already been labelled.')
        print('Stats:')
        data.fillna('not_found', inplace=True)
        # data.wn_label.fillna('not_found', inplace=True)
        # data.comb_label.fillna('not_found', inplace=True)

        # print('merriam webster:'.upper())
        # report_mw = get_metrics(data.mw_label, data.true_label)
        # print('wordnet:'.upper())
        # report_wn = get_metrics(data.wn_label, data.true_label)
        # print('dictionary.com:'.upper())
        # report_wn = get_metrics(data.dc_label, data.true_label)
        # print('combined:'.upper())
        # report_comb = get_metrics(data.comb_label, data.true_label)

        print('merriam webster:\n', classification_report(data.true_label, data.mw_label))
        print('wordnet:\n', classification_report(data.true_label, data.wn_label))
        print('dictcom:\n', classification_report(data.true_label, data.dc_label))
        print('combined:\n', classification_report(data.true_label, data.comb_label))

    data = data.sort_values('word')
    data.to_csv(path_or_buf=args.data, index=False)
    print('Data were saved.')
