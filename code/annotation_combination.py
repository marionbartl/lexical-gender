# COMPARISON OF MINE AND SUSAN'S (AND RYAN's) LABELS + majority vote

# read everything in and make sure it is ordered to avoid labeling mistakes
import numpy as np
import pandas as pd

from utils.general import conflict_resolution_3


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

    # import annotation files and sort by word to make sure they are all in the same order
    wiki_a1 = pd.read_csv('results/gendered_nouns_wiki1000_sample_a1.csv', header=0)
    wiki_a1 = wiki_a1.sort_values('word').reset_index(drop=True)

    wiki_a2 = pd.read_csv('results/gendered_nouns_wiki1000_sample_a2.csv', header=0)
    wiki_a2 = wiki_a2.sort_values('word').reset_index(drop=True)

    wiki_a3 = pd.read_csv('results/gendered_nouns_wiki1000_sample_a3.csv', header=0)
    wiki_a3 = wiki_a3.sort_values('word').reset_index(drop=True)

    wiki = pd.read_csv('results/gendered_nouns_wiki1000_sample.csv', header=0)
    wiki = wiki.sort_values('word').reset_index(drop=True)

    # check that no rows got lost anywhere in the process
    assert len(wiki_a1) == len(wiki_a2) == len(wiki_a3) == len(wiki)

    labels = ['masc', 'fem', 'neutral']
    l_mat = []  # prepare label matrix for fleiss' kappa computation
    wiki['true_label'] = None  # prepare the column with majority vote label

    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    # use the same conflict resolution function from combined label to get the majority vote over
    # all annotator labels
    for idx, row in wiki_a1.iterrows():
        ml, sl, rl = row.true_label, wiki_a2.true_label[idx], wiki_a3.true_label[idx]
        # print(ml, sl, rl, ':', conflict_resolution_3(ml, sl, rl))
        wiki['true_label'][idx] = conflict_resolution_3(ml, sl, rl)
        # majority_labels.append(conflict_resolution_3(ml, sl, rl))
        label_counts = [0 for i, _ in enumerate(labels)]
        label_counts[label2id[ml]] += 1
        label_counts[label2id[sl]] += 1
        label_counts[label2id[rl]] += 1
        l_mat.append(label_counts)

    # compute fleiss's kappa
    label_mat = np.array(l_mat)
    fk = fleiss_kappa(label_mat)
    print('Fleiss\'s kappa:', round(fk, 3))

    wiki.to_csv('results/gendered_nouns_wiki1000_sample_majority.csv', index=False)

    # Inter-annotator disagreements:
    disagree = []

    for i, row in wiki_a1.iterrows():
        assert row.word == wiki_a2.word[i] == wiki_a3.word[i]
        sl = wiki_a2.true_label[i]
        rl = wiki_a3.true_label[i]
        if row.true_label != sl or row.true_label != rl:
            disagree.append({'word': row.word, 'marion': row.true_label, 'susan': sl, 'ryan': rl})
            # print('word: {}\t marion: {}\t susan: {}\t ryan: {}'.format(row.word, row.true_label, sl, rl))

    disagree = pd.DataFrame(disagree)
    disagree.to_csv('results/inter_annotator_disagreements.csv', index=False)
