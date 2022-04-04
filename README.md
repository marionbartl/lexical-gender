# Lexical Gender Detection with Online Lexical Databases

## Find out the gender of a given word
```
from utils.dict_utils import check_dictionary

check_dictionary('babysitter', 'mw')
>> 'neutral'
```

```
python3 code/lexical_gender.py --test
```

## Data

### Gold Standard
Our _gold standard_ is a collection of 134 nouns, which we use to measure performance and
find the optimal parameter values for the overall method via grid search. 

The code below loads the gold standard data, transforms it into long format, performs a grid search and 
labels the words for gender using our three dictionaries as well as the combined method.
```
python3 code/lexical_gender.py --gold data/gendered_nouns_gold_standard.csv 
```

### Random Wikipedia Dataset

#### Download
We extracted our random corpus of 1,000 Wikipedia articles using the `wiki_corpus.py` script. 
`--n` specifies the number of articles that are being retrieved and `--file` specifies the 
JSON file in which they are being saved.
```
python3 code/wiki_corpus.py --n 1000 --file wikicorpus1000.json
```

#### Gendered Noun Extraction
We then use the `lexical_gender.py` script to extract gendered nouns from the random wikipedia corpus.
The extracted nouns are saved in the file specified after the `--file` argument.
```
python3 code/lexical_gender.py --wiki data/wikicorpus1000.json --file results/gendered_nouns_wiki1000.csv
```

Then, the extracted are filtered to only keep those for which any of the dictionary definitions resulted in 
a masculine or feminine label. 
The filtered list, _Wiki1000-sample_, is saved in `data/gendered_nouns_wiki1000_sample.csv`.


### Quantitative
```

```

