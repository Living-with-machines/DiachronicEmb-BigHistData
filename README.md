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
- time to train a model on **27,021,377** sentences: **59.4 mins**
- tokenization + training: **87.13 mins**

**Pipeline2**: 
- TBD

### Post-merging of vectors
> NB This is still experimental.
The script `merge_errors.py` takes a w2v model as input and returns a list of vectors with keys merged if they are considered spelling variants by the package `SpellChecker`.

To load a model, extract its keys, check the spelling for 356,429 words and correct the wrong ones it took: **XXX minutes**

### Comparison between models trained on different OCR qualities
Consider the following results, one from a 1860s model with OCR quality == all, the other only > 0.90.

#### Most similar to _machine_: 
**All OCR qualities**:
[('machines', 0.8097250461578369), ('maehine', 0.7315937280654907), ('sewing', 0.6739287376403809), ('mschines', 0.6591671705245972), ('machined', 0.6497806906700134), ('stitcii', 0.6432080268859863), ('machinea', 0.642343282699585), ('maciiine', 0.63560551404953), ('maciiines', 0.6317011713981628), ('machinf', 0.6287257671356201)]

**OCR>0.90**:
[('machines', 0.8224047422409058), ('maehine', 0.7288569808006287), ('sewing', 0.7196574807167053), ('machined', 0.6894606351852417), ('maciiine', 0.6764348149299622), ('stitc', 0.6748238205909729), ('achines', 0.6676322817802429), ('maohine', 0.6472572684288025), ('mortising', 0.646981418132782), ('threshing', 0.6202985048294067)]

#### Most similar to _machines_: 
**All OCR qualities**:
[('machine', 0.8097251057624817), ('machined', 0.7469484210014343), ('sewing', 0.7378761768341064), ('machinea', 0.7249653935432434), ('maehines', 0.7194136381149292), ('maehine', 0.7114378213882446), ('maohines', 0.7110329866409302), ('maciiines', 0.7070027589797974), ('achines', 0.7005993723869324), ('mschines', 0.6788889169692993)]

**OCR>0.90**:
[('machine', 0.822404682636261), ('sewing', 0.7372316718101501), ('machined', 0.7232564687728882), ('achines', 0.6847306489944458), ('maehine', 0.6781646609306335), ('stitc', 0.6730467081069946), ('maciiine', 0.6573492884635925), ('stitch', 0.6254497170448303), ('slotting', 0.619963526725769), ('achine', 0.6173325777053833)]


#### Most similar to _machinery_: 
**All OCR qualities**:
[('pitwork', 0.6965991258621216), ('machieery', 0.684580385684967), ('mashinery', 0.6752263307571411), ('maohinery', 0.6704638600349426), ('machisery', 0.6614018082618713), ('dynamometers', 0.656417727470398), ('maehinery', 0.6512506008148193), ('engines', 0.638211190700531), ('machineiy', 0.6372394561767578), ('ropemaking', 0.6365615725517273)]

**OCR>0.90**:
[('engines', 0.6259132027626038), ('ropemaking', 0.6172173023223877), ('dynamometers', 0.6092963218688965), ('machiner', 0.6085454225540161), ('gear', 0.602767825126648), ('recoating', 0.5817556977272034), ('machined', 0.5795108079910278), ('maehinery', 0.5749990940093994), ('reshipping', 0.5673781633377075), ('carding', 0.5659915208816528)]

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