import re


def get_root(phrase, nlp):
    """extract root noun from nounphrase"""
    # print('Let\'s find the main noun in the NP "' + NP + '"!')
    return [token.text for token in nlp(phrase) if token.dep_ == 'ROOT'][0]


def singular_referent_gender(referent, gendered_words):
    # check if the referent is an explicitly gendered noun ...
    if gendered_words.masculine.str.contains(referent).any():
        ref_morph = 'masc'
    elif gendered_words.feminine.str.contains(referent).any():
        ref_morph = 'fem'
    else:  # ... or not
        ref_morph = None

    return ref_morph


def suffix_heuristics(word):
    man_end = re.compile(r'(?<!wo|hu)m[ae]n$')
    woman_end = re.compile(r'wom[ae]n$')
    girl_end = re.compile(r'girls*$')
    boy_end = re.compile(r'boys*$')

    if man_end.search(word) or boy_end.search(word):
        return 'masc'
    elif woman_end.search(word) or girl_end.search(word):  # or word.endswith('ess'):
        return 'fem'
    else:
        return


def prefix_heuristics(word):
    man_start_dash = re.compile(r'^(?<!wo)m[ae]n-')
    woman_start_dash = re.compile(r'^wom[ae]n-')

    if man_start_dash.search(word):
        return 'masc'
    elif woman_start_dash.search(word):
        return 'fem'
    else:
        return
