import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def conflict_resolution_2(a, b):
    """this function deals with conflict between labels a and b.
    If they are not the same and one is not_found, the more definite label is used as combined label.
    If they are not the same and one is neutral, the neutral label is used as combined label.
    Possible labels include: masc, fem and neutral"""

    gendered = ['fem', 'masc']
    labels = ['fem', 'masc', 'neutral']

    if a == b:
        resolved = b
    elif a != b:
        if a in gendered and b not in gendered:
            if b == 'neutral':
                resolved = b
            else:
                resolved = a
        elif b in gendered and a not in gendered:
            if a == 'neutral':
                resolved = a
            else:
                resolved = b
        elif (not a or a == 'not_found') and b in labels:
            resolved = b
        elif (not b or b == 'not_found') and a in labels:
            resolved = a
        else:  # case in which both a and b are gendered
            resolved = 'neutral'
    else:
        resolved = 'neutral'
    # if a != b:
    #     print(a, b)
    #     print('resolved as: ', resolved)
    return resolved


def conflict_resolution_3(a, b, c):
    """this function deals with conflict between labels a, b, c.
    Possible labels include: masc, fem and neutral"""

    if a == b == c:
        return b
    count_labels = Counter([a, b, c])
    if max(count_labels.values()) >= 2:
        max_label = max(count_labels, key=count_labels.get)
        if max_label and max_label != 'not_found':
            return max_label
        else:
            return min(count_labels, key=count_labels.get)
    else:
        if ('not_found' in count_labels.keys() or None in count_labels.keys()) and \
                (count_labels['not_found'] == 1 or count_labels[None] == 1):
            # the counts of the other two must be 1 too
            two_specific = list(count_labels.keys())
            two_specific.remove('not_found')
            # print('one is not_found, the others are:', two_specific)
            return conflict_resolution_2(two_specific[0], two_specific[1])
        else:
            return 'neutral'


def is_word(word):
    # https://en.wiktionary.org/wiki/Category:English_one-letter_words
    if len(word) == 1 and word not in ['I', 'a', 'A', 'O', 'o']:
        return False
    non_word_char = re.compile(r'[^a-zA-Z-]')
    if non_word_char.search(word):
        return False
    else:
        if word == '-':
            return False
        return True


def get_metrics(pred_list, labels, remove_na=False):
    assert len(pred_list) == len(labels)
    if remove_na:
        pred_list.dropna(inplace=True)
    else:
        pred_list.fillna('not_found', inplace=True)

    le = LabelEncoder()

    y = list(zip(labels, pred_list))

    y_pred = le.fit_transform([p for t, p in y])
    y_true = le.transform([t for t, p in y])

    print(le.classes_)

    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
    label2name = {lab: name for lab, name in enumerate(le.classes_)}
    report = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)

    print('Confusion Matrix')
    conf_mat = confusion_matrix(y_true, y_pred)
    cf = np.column_stack((le.classes_, conf_mat))
    cf = np.vstack((np.array([''] + list(le.classes_)), cf))
    cf_df = pd.DataFrame(cf)
    print(cf_df.to_string())
    # sns.heatmap(conf_mat / np.sum(conf_mat), annot=True, # fmt='.2%',
    #             xticklabels=le.classes_, yticklabels=le.classes_)

    group_counts = [str(value) for value in conf_mat.flatten()]
    group_percentages = ['{0:.2f}%'.format(value) if value > 0 else '' for value in
                         conf_mat.flatten() / np.sum(conf_mat) * 100]

    cm_labels = [f'{v1}\n{v2}' if v2 else f'{v1}' for v1, v2 in zip(group_counts, group_percentages)]
    cm_labels = np.asarray(cm_labels).reshape(conf_mat.shape)

    classes = [l.replace('_', '\n') for l in le.classes_]

    sns.set(font_scale=1.3)

    sns.heatmap(conf_mat, annot=cm_labels, fmt='', cbar=False,
                xticklabels=classes, yticklabels=classes, cmap='Greys',
                robust=True, linewidths=0.2, linecolor='darkgrey')
    plt.xlabel('predicted labels')
    plt.ylabel('true labels')
    plt.show()
    print()

    return report, label2name
