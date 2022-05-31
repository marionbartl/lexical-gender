# Lexical Gender Detection with Online Lexical Databases
We present the code associated with our paper 
"Inferring Gender: A Scalable Methodology for Gender Detection with Online Lexical Databases", which was 
accepted for publication at the Second Workshop on Language Technology for Equality, Diversity, Inclusion (LT-EDI-2022)
at the ACL Conference 2022 in Dublin. 

```
@inproceedings{bartl-leavy-2022-inferring,
    title = "Inferring Gender: A Scalable Methodology for Gender Detection with Online Lexical Databases",
    author = "Bartl, Marion  and
      Leavy, Susan",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.ltedi-1.7",
    pages = "47--58",
    abstract = "This paper presents a new method for automatic detection of gendered terms in large-scale language datasets. Currently, the evaluation of gender bias in natural language processing relies on the use of manually compiled lexicons of gendered expressions, such as pronouns and words that imply gender. However, manual compilation of lists with lexical gender can lead to static information if lists are not periodically updated and often involve value judgements by individual annotators and researchers. Moreover, terms not included in the lexicons fall out of the range of analysis.To address these issues, we devised a scalable dictionary-based method to automatically detect lexical gender that can provide a dynamic, up-to-date analysis with high coverage. Our approach reaches over 80{\%} accuracy in determining the lexical gender of words retrieved randomly from a Wikipedia sample and when testing on a list of gendered words used in previous research.",
}

```

## The Basics
### The basic function

The `check_dictionary` functions takes a word and the name of a dictionary and returns the word's lexical gender label
based on the presence of gendered words in its dictionary definition.
Additional parameters are 
- `heuristics`, which determines whether to use suffix information of the given word, 
- `seed_pairs`, the number of gendered word pairs to search for in the dictionary definition, 
- `no_words`, the maximum number of words to use from each definition, and
- `no_defs`, the maximum number of definitions to use from the word's page in the dictionary.


```python
from utils.dict_utils import check_dictionary

check_dictionary(word='babysitter', 
                 dict_abbrev='merriam', 
                 heuristics=True, 
                 seed_pairs=5, no_words=35, no_defs=10)
```
```
>> 'neutral'
```
### Test the method
If you want to try out the lexical gender detection, run the `lexical_gender.py` script with the `--test` argument
and optional arguments that contain the model parameters.
```commandline
python3 code/lexical_gender.py --test [--heur] [--n_words N_WORDS] [--s_pairs S_PAIRS] [--n_defs N_DEFS]
```
Three words will be tested and the output will look similar to this: 
```
>> Test word: contraceptive
Merriam Webster-label: neutral
Wordnet-label: neutral
Dictionary.com-label: neutral
combined label: neutral
```

## Data

### Gold Standard
Our _gold standard_ is a collection of 134 nouns, which we use to measure performance and
find the optimal parameter values for the overall method via grid search. 

The command below loads the gold standard data, transforms it into long format, performs a grid search, and subsequently
labels the words in the gold standard corpus for lexical gender.
```commandline
python3 code/lexical_gender.py --gold data/gendered_nouns_gold_standard.csv 
```

### Random Wikipedia Dataset

#### Download
We extracted our random corpus of 1,000 Wikipedia articles using the `wiki_corpus.py` script. 
`--n` specifies the number of articles that are being retrieved and `--file` specifies the 
JSON file in which they are being saved.
```commandline
python3 code/wiki_corpus.py --n 1000 --file data/wikicorpus1000.json
```

#### Gendered Noun Extraction
We then use the `lexical_gender.py` script to extract gendered nouns from the random wikipedia corpus.
The extracted nouns are saved in the file specified after the `--file` argument.
```commandline
python3 code/lexical_gender.py --wiki data/wikicorpus1000.json --file results/gendered_nouns_wiki1000.csv
```

Then, the extracted nouns are filtered to only keep those for which any of the dictionary definitions resulted in 
a masculine or feminine label. 
The filtered list of nouns, **Wiki1000-sample**, is saved in `data/gendered_nouns_wiki1000_sample.csv`.


## Evaluation
For the description and evaluation of our method on all datasets, the `lexical_evaluation.py` script is used.
The output of the script is divided into three sections: 
1. Label count of gold standard, Wiki1000-sample and the full Wiki1000 (Table 2 in the original paper)
2. Performance evaluation of the method on the gold standard and the Wiki1000 sample
3. Overlap analysis between Wiki1000-sample and gold standard
```commandline
python3 code/lexical_gender_evaluation.py
```



Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

