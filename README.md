# Word embeddings for historical newspaper corpora

Table of contents
-----------------

- [Repository structure](#repository-structure)
- [Pre-processing](#pre-processing)
- [Tokenization and training](#tokenization-and-training)
- [Merging of vectors for OCR errors](#merging-of-vectors-for-ocr-errors)
- [Processing times](#processing-times)
    * [Pre-processing](#pre-processing)
    * [Tokenization and training](#tokenization-and-training)
    * [Merging of vectors (OCR errors post-correction)](#merging-of-vectors-post-correction-of-ocr-errors)
- [Comparison between models trained on different OCR qualities](#comparison-between-models-trained-on-different-ocr-qualities)
    * [Most similar to machine](#most-similar-to-machine)
    * [Most similar to machines](#most-similar-to-machines)
    * [Most similar to machinery](#most-similar-to-machinery)
- [Notes](#notes)
    * [Bottlenecks and decisions made](#bottlenecks-and-decisions-made)
    * [Decisions to be made](#decisions-to-be-made)
- [References and relevant literature](#references-and-relevant-literature)

## Repository structure
```
newspaper_embedding_models
├─ README.md
└─ scripts
   ├─ alignment
   │    ├─ align-procrustes-from-vectors.py
   │    └─ align-procrustes.py
   ├─ exploration
   │    └─ word_diachrony.ipynb
   ├─ postprocess
   │    └─ merge_errors.py
   ├─ preprocess
   │    ├─ listpathsbyyear.py
   │    └─ preprocess=alto2txt-totxt.py
   └─ training
        └─ train_fromtxt.py
```

## Pre-processing
> Script: `preprocess-alto2txt-totxt.py`
- Run `listpathsbyyear.py` to generate one file per decade containing a list of directories in Azure containing the articles for that decade.
- Read alto2txt-ed plain text files decade by decade (from the output of `listpathsbyyear.py`).
- Remove newlines (due to the OCR)
- Divide by sentences (`nltk.tokenize.sent_tokenize`)
- Lowercase
- Remove punctuation
- Remove stopwords (`nltk.stopwords.words("english")`)
- Write sentences in one txt file per decade (named XXXXs.txt, e.g. 1880s.txt)


## Tokenization and training
> Script: `train_fromtxt.py`
> Note: new models will be released soon after a grid search through some of the main parameters (optimal ones: `min_count`=1, `window`=3, `epochs`=5, `sg`=True, `vector_size`=200)
- For each txt file for a decade:
    * Initialize empty list (which will contain all tokenized sentences to be used for training)
    * Tokenize sentences and add tokens to the list (not the sentences: this will kill the process, as we have too many sentences)
    * Randomly initialize w2v model (params: `min_count`=10, `window`=5, `epochs`=5, `sg`=True, `vector_size`=200)
    * Train model on decade
    * Save model as `XXXXs.model` (e.g. `1860s.model`)


## Processing times
### Pre-processing
> Sample: whole LWM and HMD collections
- processing times for **4.7B** words (7,102,851 articles): **7 hours and 32 minutes**

### Tokenization and training
> Sample: 1800s (all OCR qualities)
- time to tokenize **?** sentences: **? min**.
- time to train a model on **?** sentences: **? mins**
- tokenization + training: **? mins**

> Sample: 1810s (all OCR qualities)
- time to tokenize **?** sentences: **1 min**.
- time to train a model on **?** sentences: **29 mins**
- tokenization + training: **34 mins**

> Sample: 1820s (all OCR qualities)
- time to tokenize **13,495,444** sentences: **2:07 mins**.
- time to train a model on **13,495,444** sentences: **37 mins**
- tokenization + training: **41 mins**

> Sample: 1830s (all OCR qualities)
- time to tokenize **9,389,676** sentences: **01:25 mins**.
- time to train a model on **9,389,676** sentences: **27 mins**
- tokenization + training: **30 mins**

> Sample: 1840s (all OCR qualities)
- time to tokenize **30,181,001** sentences: **04:11 mins**.
- time to train a model on **30,181,001** sentences: **1h 4 mins**
- tokenization + training:  **1h 11 mins**

> Sample: 1850s (all OCR qualities)
- time to tokenize **37,786,194** sentences: **04:54 mins**.
- time to train a model on **37,786,194** sentences: **77.75 mins**
- tokenization + training: **86.36 mins**

> Sample: 1860s (all OCR qualities)
- time to tokenize **34,126,059** sentences: **13 mins**.
- time to train a model on **34,126,059** sentences: **1h 17 mins**
- tokenization + training: **1 h 50m**

> Sample: 1870s (all OCR qualities)
- time to tokenize **27,021,377** sentences: **9:55 mins**.
- time to train a model on **27,021,377** sentences: **59.4 mins**
- tokenization + training: **87.13 mins**

> Sample: 1880s (all OCR qualities)
- time to tokenize **35,716,553** sentences: **4:39 mins**.
- time to train a model on **35,716,553** sentences: **66.39 mins**
- tokenization + training: **74.47 mins**

> Sample: 1890s (all OCR qualities)
- time to tokenize **34,077,373** sentences: **03:48 mins**.
- time to train a model on **34,077,373** sentences: **52.38 mins**
- tokenization + training: **59.26 mins**

> Sample: 1900s (all OCR qualities)
- time to tokenize **23,530,425** sentences: **02:44 mins**.
- time to train a model on **23,530,425**  sentences: **32.7 mins**
- tokenization + training: **37.47 mins**

> Sample: 1910s (all OCR qualities)
- time to tokenize **14,678,780** sentences: **01:37 mins**.
- time to train a model on **34,077,373** sentences: **18.33 mins**
- tokenization + training: **21.23 mins**

### Merging of vectors (post-correction of OCR errors)
To load a model, extract its keys, check the spelling for **356,429 word vectors** (all 1870s, HMD+LWM) and merge the mispellings it took: **4.40 hours**. 
NB: 258,944 out of 356,429 words were considered mispellings! Note that out of the remaiining 97,484 words there are still certainly several mispellings, which SpellChecker did not manage to correct. 


## Merging of vectors for OCR errors
> NB: This is still experimental.
The script `merge_errors.py` takes a w2v model as input and returns a list of vectors with keys merged if they are considered spelling variants by the package `SpellChecker`. The output needs to be loaded as `KeyedVector`, not as a full `Word2Vec` model.


## Comparison between models trained on different OCR qualities
Consider the following results, one from a 1860s model with OCR quality == all, the other only > 0.90.

### Most similar to _machine_
**All OCR qualities**:
[('machines', 0.8097250461578369), ('maehine', 0.7315937280654907), ('sewing', 0.6739287376403809), ('mschines', 0.6591671705245972), ('machined', 0.6497806906700134), ('stitcii', 0.6432080268859863), ('machinea', 0.642343282699585), ('maciiine', 0.63560551404953), ('maciiines', 0.6317011713981628), ('machinf', 0.6287257671356201)]

**OCR>0.90**:
[('machines', 0.8224047422409058), ('maehine', 0.7288569808006287), ('sewing', 0.7196574807167053), ('machined', 0.6894606351852417), ('maciiine', 0.6764348149299622), ('stitc', 0.6748238205909729), ('achines', 0.6676322817802429), ('maohine', 0.6472572684288025), ('mortising', 0.646981418132782), ('threshing', 0.6202985048294067)]

### Most similar to _machines_
**All OCR qualities**:
[('machine', 0.8097251057624817), ('machined', 0.7469484210014343), ('sewing', 0.7378761768341064), ('machinea', 0.7249653935432434), ('maehines', 0.7194136381149292), ('maehine', 0.7114378213882446), ('maohines', 0.7110329866409302), ('maciiines', 0.7070027589797974), ('achines', 0.7005993723869324), ('mschines', 0.6788889169692993)]

**OCR>0.90**:
[('machine', 0.822404682636261), ('sewing', 0.7372316718101501), ('machined', 0.7232564687728882), ('achines', 0.6847306489944458), ('maehine', 0.6781646609306335), ('stitc', 0.6730467081069946), ('maciiine', 0.6573492884635925), ('stitch', 0.6254497170448303), ('slotting', 0.619963526725769), ('achine', 0.6173325777053833)]


### Most similar to _machinery_
**All OCR qualities**:
[('pitwork', 0.6965991258621216), ('machieery', 0.684580385684967), ('mashinery', 0.6752263307571411), ('maohinery', 0.6704638600349426), ('machisery', 0.6614018082618713), ('dynamometers', 0.656417727470398), ('maehinery', 0.6512506008148193), ('engines', 0.638211190700531), ('machineiy', 0.6372394561767578), ('ropemaking', 0.6365615725517273)]

**OCR>0.90**:
[('engines', 0.6259132027626038), ('ropemaking', 0.6172173023223877), ('dynamometers', 0.6092963218688965), ('machiner', 0.6085454225540161), ('gear', 0.602767825126648), ('recoating', 0.5817556977272034), ('machined', 0.5795108079910278), ('maehinery', 0.5749990940093994), ('reshipping', 0.5673781633377075), ('carding', 0.5659915208816528)]

## Notes
### Bottlenecks and decisions made
- List comprehensions are _much_ slower than appending to a list with each iteration.
- `csv` is much faster than `pandas` when looping through CSV files.
- Either way, iterating over lines is much faster than working with columns as lists.
- Punctuation removal and lowercasing are _fast_ and should be easy to scale up to a much larger corpus.
- Tokenization and stopword removal were the main bottleneck, but tokenization using NLTK's `regexp` makes it decently fast.
- _Not caching_ is the real bottleneck with stopword removal. First cache the stopwords in a list, then remove them.
- Removing stopwords from a string is _much_ faster than from a list (i.e. first remove stopwords, then tokenize).
- OCR'd text has line breaks marked with hyphen. These need to be removed before anything else.

### Decisions to be made
- Stopwords: use a precompiled one (e.g. `nltk`) or compile our own? Currently using nltk.
- Do we want to remove tokens with less then _n_ characters (_n_ to be agreed upon)? Currently not doing so.
- Articles with very low OCR quality. The few articles pre-1830 in the LWM collection, for instance, have such a low OCR quality that there are 'no words' occurring more than 5 times. How do we deal with these?
- Do we want to align using Procrustes or do we want to simply initialize the embedding of a decade with the vectors from the model for the previous decade each time? See [[1]](#1) for discussion. Currently using Procrustes.
- If Procrustes, generalized orthogonal or pairwise? And which method in terms of source model (model to align to)? Currently using Orthogonal and aligning each decade to the previous one.
- When sentences are collected to train a word2vec model, they are currently stored in a list. Can less memory-consuming types be used? Currently models trained on more than 15M articles need at least 32GiB of RAM.
- Spell check. `spell_checker.py` works well, but it seems time consuming (to be checked systematically). Currently applying it post-training by merging vectors of misspellings.


## References and relevant literature
> This list is not exhaustive.

<a id="1">[1]</a> 
Tsakalidis, A., Basile, P., Bazzi, M. et al. DUKweb, diachronic word representations from the UK Web Archive corpus. Sci Data 8, 269 (2021). https://doi.org/10.1038/s41597-021-01047-x

<a id="2">[2]</a> 
Yoon Kim, Yi-I Chiu, Kentaro Hanaki, Darshan Hegde, and Slav Petrov. 2014. Temporal Analysis of Language through Neural Language Models. In Proceedings of the ACL 2014 Workshop on Language Technologies and Computational Social Science, pages 61–65, Baltimore, MD, USA. Association for Computational Linguistics. http://dx.doi.org/10.3115/v1/W14-2517

<a id="3">[3]</a> 
Philippa Shoemark, Farhana Ferdousi Liza, Dong Nguyen, Scott Hale, and Barbara McGillivray. 2019. Room to Glo: A Systematic Comparison of Semantic Change Detection Approaches with Word Embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 66–76, Hong Kong, China. Association for Computational Linguistics. http://dx.doi.org/10.18653/v1/D19-1007

<a id="4">[4]</a> 
William L. Hamilton, Jure Leskovec, and Dan Jurafsky. 2016. Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1489–1501, Berlin, Germany. Association for Computational Linguistics. http://dx.doi.org/10.18653/v1/P16-1141

<a id="5">[5]</a> 
Andrey Kutuzov, Lilja Øvrelid, Terrence Szymanski, and Erik Velldal. 2018. Diachronic word embeddings and semantic shifts: a survey. In Proceedings of the 27th International Conference on Computational Linguistics, pages 1384–1397, Santa Fe, New Mexico, USA. Association for Computational Linguistics. https://aclanthology.org/C18-1117