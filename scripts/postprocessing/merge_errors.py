#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
--------------------
Merges OCR errors in word embedding models
--------------------
Given a model for one decade it applies SpellChecker to the whole vocabulary, 
merging vectors that are considered to be mispellings with the 'correct' 
word vector by summing and averaging the vector

How to run:
    $ python train_diach_emb.py

Variables to specify:
    modelpath (str): path to the model (.model) file on which to run the spellchecker
    newmodelname (str): name to be given to the new vector file

Returns:
    .txt file containing the modified vector space. By default, it will be saved in the same directory as the original model.
"""

from gensim.models import Word2Vec
from spellchecker import SpellChecker
from tqdm import tqdm
from collections import defaultdict

spell = SpellChecker()

# ------------------- Specify path to the model and name to be given to new vector file --------------------

modelpath = './outputs/lwmhmdtest/raw/time2.model'
newmodelname = 'time2-corrected-vectors.txt'
basepath = '/'.join(modelpath.split('/')[:-1])

# ------------------- Load model --------------------

model = Word2Vec.load(modelpath)

# ------------------- Extract all keys --------------------

vectors = model.wv.key_to_index

# ------------------- Initialize dict to contain spelling variants --------------------

allwords = defaultdict(dict)

for vec in tqdm(vectors):
    item2 = spell.correction(vec)
    if item2 != vec:
        allwords[vec] = item2

# ------------------- For each spelling variant, do the average with the correct spelling --------------------

for k,v in tqdm(allwords.items()):
    if k != v:
        try:
            model.wv[v] = (model.wv[v] + model.wv[k]) / 2
        except KeyError:
            continue

# ------------------- Save vector file only --------------------

model.wv.save_word2vec_format('{}/{}'.format(basepath,newmodelname), binary=False)