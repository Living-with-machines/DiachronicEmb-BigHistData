# Word embeddings for historical newspaper corpora

Table of contents
-----------------

- [Repository structure](#repository-structure)
- [Pre-processing](#pre-processing)
    * [Pipeline1](#pipeline1)
    * [Pipeline2](#pipeline2)
- [Tokenization and training](#tokenization-and-training)
    * [Pipeline1](#pipeline1)
    * [Pipeline2](#pipeline2)
- [Processing times](#processing-times)
    * [Pre-processing](#pre-processing)
    * [Tokenization and training](#tokenization-and-training)
- [Notes](#notes)
    * [Bottlenecks and decisions made](#bottlenecks-and-decisions-made)
    * [Decisions to be made](#decisions-to-be-made)
- [References](#references)


## Repository structure
```
newspaper_embedding_models
├─ README.md
├─ input_data
│  └─ test1
│     ├─ 1860s.txt
│     └─ 1870s.txt
├─ outputs
│  └─ test1
│     ├─ 1860s.model
│     └─ 1870s.model
└─ scripts
   ├─ alignment
   │     └─ align-procrustes.py
   ├─ exploration
   │     └─ TBD
   ├─ preprocess
   │    ├─ listpathsbyyear.py
   │    ├─ preprocess=alto2txt-tocsv.py
   │    ├─ preprocess=alto2txt-totxt.py
   │    └─ spell_checker.py
   └─ training
        ├─ train_fromcsv.py
        └─ train_fromtxt.py
```

## Pre-processing
### Pipeline1
> Script: `preprocess-alto2txt-totxt.py`
- Run `listpathsbyyear.py` to generate one file per decade containing a list of directories in Azure containing the articles for that decade.
- Read alto2txt-ed plain text files decade by decade (from the output of `listpathsbyyear.py`).
- Remove newlines (due to the OCR)
- Divide by sentences (`nltk.tokenize.sent_tokenize`)
- Lowercase
- Remove punctuation
- Remove stopwords (`nltk.stopwords.words("english")`)
- Write sentences in one txt file per decade (named XXXXs.txt, e.g. 1880s.txt)

### Pipeline2
> Script: `preprocess-alto2txt-tocsv.py`
- Read one alto2txt-ed plain text file (one article) at a time
- Remove newlines (due to the OCR)
- Divide by sentences (`nltk.tokenize.sent_tokenize`)
- Lowercase
- Remove punctuation
- Remove stopwords (`nltk.stopwords.words("english")`)
- Write one sentence per row in a csv file with two variables: `year`, `text`.
> NB: It does not include division by decade: this would be done just prior to training.

## Tokenization and training
### Pipeline1
> Script: `train_fromtxt.py`
- For each txt file for a decade:
    * Initialize empty list (which will contain all tokenized sentences to be used for training)
    * Tokenize sentences, append them to the list
    * Randomly initialize w2v model (params: `min_count`=10, `window`=5, `epochs`=5, `sg`=True, `vector_size`=200)
    * Train model on decade
    * Save model as `XXXXs.model` (e.g. `1860s.model`)

### Pipeline2
> Script: `train_fromcsv.py`
- Read CSV file containing all texts for all decades
- Create list of dfs with one df per decade
- For each df:
    * Initialize empty list (which will contain all tokenized sentences to be used for training)
    * Tokenize sentences, append them to the list
    * Randomly initialize w2v model (params: `min_count`=10, `window`=5, `epochs`=5, `sg`=True, `vector_size`=200)
    * Train model on decade
    * Save model as `XXXXs.model` (e.g. `1860s.model`)

## Processing times
### Pre-processing
**Pipeline1**: 
> Sample: whole LWM and HMD collections
- processing times for **4.7B** words (7,102,851 articles): **7 hours and 32 minutes**

**Pipeline2**: 
> Sample: whole LWM collection
- processing times for **2.3B** words (4,201,295 articles): **2 hours and 33 minutes**

### Tokenization and training
**Pipeline1**: 
> Sample: 1860's
- time to tokenize **34,126,059** sentences: **13 mins**.
- time to train a model on **34,126,059** sentences: **1h 17 mins**
- tokenization + training: **1 h 50m**

> Sample: 1870's
- time to tokenize **27,021,377** sentences: **9:55 mins**.
- time to train a model on **27,021,377** sentences: **???**
- tokenization + training: **???**

**Pipeline2**: 
- TBD


## Notes
### Bottlenecks and decisions made
- List comprehensions are _much_ slower than appending to a list with each iteration.
- `csv` is much faster than `pandas` when working with CSV files.
- Either way, iterating over lines is much faster than working with columns as lists.
- Punctuation removal and lowercasing are _fast_ and should be easy to scale up to a much larger corpus.
- Tokenization and stopword removal were the main bottleneck, but tokenization using NLTK's `regexp` makes it decently fast.
- _Not caching_ is the real bottleneck with stopword removal. First cache the stopwords in a list, then remove them.
- Removing stopwords from a string is _much_ faster than from a list (i.e. first remove stopwords, then tokenize).
- Two possible output formats from pre-processing: 
    1. a df (saved as csv) containing the preprocessed texts in one column and year on another 
    2. one txt file per decade. 
    The former has the advantage that we can also easily add NLP number as another column, thus making the preprocessed texts more easily reusable if one decides to then subsample across other variables at a later stage. Txt files are faster, but they also mean that we'll have to repeat the preprocessing every time we create a subsample.
- OCR'd text has line breaks marked with hyphen. These need to be removed before anything else.

### Decisions to be made
- Stopwords: use a precompiled one (e.g. `nltk`) or compile our own?
- Do we want to remove tokens with less then _n_ characters (_n_ to be agreed upon)?
- Articles with very low OCR quality. The few articles pre-1830 in the LWM collection, for instance, have such a low OCR quality that there are 'no words' occurring more than 5 times. How do we deal with these?
- Do we want to align using Procrustes or do we want to simply initialize the embedding of a decade with the vectors from the model for the previous decade each time? See [[1]](#1) for discussion.
- If procrustes, generalized orthogonal or pairwise? And which method in terms of source model (model to align to)?
- When sentences are collected to train a word2vec model, they are currently stored in a list. Consider less memory-consuming types?
- Spell check. `spell_checker.py` works well, but it seems time consuming (to be checked systematically).


## References
<a id="1">[1]</a> 
Tsakalidis, A., Basile, P., Bazzi, M. et al. DUKweb, diachronic word representations from the UK Web Archive corpus. Sci Data 8, 269 (2021). https://doi.org/10.1038/s41597-021-01047-x

<a id="2">[2]</a> 
Kim, Y., Chiu, Y.I., Hanaki, K., Hegde, D. & Petrov, S. Temporal Analysis of Language through Neural Language Models. Proceedings
of the ACL 2014 Workshop on Language Technologies and Computational Social Science, pp. 61–65 (2014).

<a id="3">[3]</a> 
Shoemark, P., Ferdousi, L.F., Nguyen, D., Scott, H. & McGillivray, B. Room to Glo: A systematic comparison of semantic change
detection approaches with word embeddings. Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 66–76 (2019).