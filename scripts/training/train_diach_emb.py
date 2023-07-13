#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
--------------------
Main code for diacronic embedding preprocessing, training and alignment.
--------------------

How to run:
    $ python train_diach_emb.py

Before running this script, you need to:
    - have all the text representing each time slice in separate .txt files, 
      where each line is a smaller chunk than the whole text 
      (depending on your task, you may want to divide it into sentences, articles,
      paragraphs, etc.).
    - define all bespoke variables in the config.yaml file. These include the path to
      the folder containing all the texts, named fittingly to represent the time slice.
      NB: the names of each file will be used to recognize the last time period, so
      make sure the names can be ordered automatically (e.g. time1, time2, ..., time10, 
      where time10 will be automatically considered as the most recent time slice).
      See docs for the full list of customizable variables.

Returns:
    outputs/ (dir): ./outputs/ folder where all models will be saved.
    outputs/test_run_name/ (dir): Dedicated folder under outputs/ where all outputs 
                                  from a specific test run will be saved. This is also where 
                                  preprocessed texts will be saved by default.
    outputs/test_run_name/raw/ (dir): Dedicated folder under outputs/test_run_name/ for all raw (non-aligned) w2v models.
    outputs/test_run_name/aligned/ (dir): Dedicated folder under outputs/test_run_name/ for all aligned w2v models.
"""

import os
from glob import glob
import yaml
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from gensim.models import Word2Vec
from tqdm import tqdm
import timeit
from utils import stopwdsrm, cleantxt, smart_procrustes_align_gensim

# ------------------- Start timing the whole process --------------------

startall = timeit.default_timer()


# ------------------- Import configs --------------------

with open("./scripts/training/config.yaml", "r") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

# --- Relevant vars
## -- General
namethetest = configs['namethetest']
inputfiles = configs['inputs']['directory']
skip_preprocessing = configs['preprocessing']['skip']
skip_training = configs['training']['skip']
skip_alignment = configs['alignment']['skip']

## -- Preprocessing
savepreprocessed = configs['preprocessing']['savepreprocessed']
minwordlength = configs['preprocessing']['pipelines']['minwordlength']
lowercase = configs['preprocessing']['pipelines']['lowercase']
remove_punctuation = configs['preprocessing']['pipelines']['remove_punctuation']
remove_stopwords = configs['preprocessing']['pipelines']['remove_stopwords']

## -- Training
epochs = configs['training']['options']['epochs']
vector_size = configs['training']['options']['vector_size']
sg = configs['training']['options']['sg']
min_count = configs['training']['options']['min_count']
window = configs['training']['options']['window']
start_alpha = configs['training']['options']['start_alpha']
end_alpha = configs['training']['options']['end_alpha']
workers = configs['training']['options']['workers']

# ------------------- Create dirs if needed --------------------

# --- General outputs directory
if not os.path.exists('./outputs'):
	os.mkdir('./outputs')

# --- Dir for these specific models
if not os.path.exists('./outputs/{}'.format(namethetest)):
	os.mkdir('./outputs/{}'.format(namethetest))

# --- Place non-aligned embeddings in a dedicated dir
if not os.path.exists('./outputs/{}/raw'.format(namethetest)):
	os.mkdir('./outputs/{}/raw'.format(namethetest))

# --- Place aligned embeddings in a dedicated dir
if not os.path.exists('./outputs/{}/aligned'.format(namethetest)):
	os.mkdir('./outputs/{}/aligned'.format(namethetest))

# --- Save preprocessed texts in a dedicated dir
if savepreprocessed == True:
    if not os.path.exists('./outputs/{}/preprocessed_corpus'.format(namethetest)):
        os.mkdir('./outputs/{}/preprocessed_corpus'.format(namethetest))

# ------------------- Preprocessing & training --------------------

# --- List the paths for all the texts inside the input dir
alltexts = glob(f'{inputfiles}/*txt')

# --- Define stopwords
cachedStopWords = stopwords.words("english")

## -- To expand the default NLTK stopwords for English, uncomment the following three lines and add your own:
#newstopwords = ['mrs','mr','say','says','said','tell','told','seem','seems','seemed','ask','asks','asked','upon','aboard','about','above','account','across','after','against','ago','ahead','along','alongside','amid','among','around','aside','at','atop','away','because','before','behalf','behind','below','beneath','beside','besides','between','beyond','but','by','circa','considering','depending','despite','down','due','during','for','from','further','given','in','including','inside','instead','into','less','like','near','notwithstanding','of','off','on','onto','opposite','other','out','outside','over','owing','per','plus','regarding','regardless','round','since','than','thanks','through','throughout','till','to','together','toward','towards','under','underneath','unlike','until','up','upon','versus','via','with','within','without']
#cachedStopWords.extend(newstopwords)
#cachedStopWords = list(dict.fromkeys(cachedStopWords))

for subcorpus in alltexts:
    # --- Start timing for the specific time slice
    startslice = timeit.default_timer()

    # --- Get name of the time slice (i.e. the last bit of the file path without the ext)
    nameoftimeslice = subcorpus.split('/')[-1].split('.')[0]

    print(f'Now preprocessing {nameoftimeslice}...')

    # --- Open new file in w mode if option to save the preprocessed text is chosen
    if savepreprocessed == True:
        newfile = open(f'./outputs/{namethetest}/preprocessed_corpus/{nameoftimeslice}.txt', 'w')
    
    # --- Start list of sentences (i.e. articles, text chunks, etc., depending on input), this will be input of w2v
    sentences = []

    with open(subcorpus) as infile:
        # --- Read line by line and preprocess
        for line in tqdm(infile):
            ## -- Preprocess the texts according to configs
            text = cleantxt(line, remove_punctuation, lowercase)
            ## -- Remove stopwords if chosen to
            if remove_stopwords == True:
                text = stopwdsrm(text, cachedStopWords, minwordlength) # Remove stopwords
            ## -- Write the line to the newfile if chosen to (without tokenizing)
            if savepreprocessed == True:
                if text != ' ' and text != '' and text != '\n':
                    newfile.write(text + '\n')
            ## -- Tokenize if what's left after preprocessing is not just whitespace
            if skip_training == False:
                if text != ' ' and text != '' and text != '\n':
                    text = regexp_tokenize(text, pattern='\s+', gaps =True)
                    sentences.append(text)
    
    print('All preprocessing finished. Altogether it took {} minutes.'.format((timeit.default_timer() - startslice) / 60))
    
    if skip_training == False:
        # --- Initialize w2v model with selected parameters
        w2v_model = Word2Vec(min_count=min_count,
                            window=window, 
                            sg=sg,
                            vector_size=vector_size,
                            workers=workers)

        # --- Build vocabulary from the sentences based on the initialized model
        print('Building the vocab...')
        w2v_model.build_vocab(sentences)
        print('Vocab built!')
        
        # --- Start the training
        print('Now training the {}s model... This might take a very long time. Have a coffee. Or two.'.format(nameoftimeslice))
        
        # --- Start timing for training process
        traintime = timeit.default_timer()
        
        w2v_model.train(sentences, 
                        total_examples=w2v_model.corpus_count, 
                        start_alpha=start_alpha, 
                        end_alpha=end_alpha, 
                        epochs=epochs)
        
        # --- Save model
        w2v_model.save("./outputs/{}/raw/{}.model".format(namethetest,nameoftimeslice))

        stopslice = timeit.default_timer()
        print('{} done! It took {} minutes'.format(nameoftimeslice, (stopslice - startslice)/60))

# ------------------- Alignment --------------------

if skip_alignment == False:
    print('Now starting alignement with Orthogonal Procrustes...')
    # --- Start timing alignment process
    startalign = timeit.default_timer()

    # --- List all raw models and order them by name
    allmodels = sorted(glob('./outputs/{}/raw/*model'.format(namethetest)))

    # --- Load last model (fixed time slice)
    model1 = Word2Vec.load(allmodels[-1])

    # --- Just get the model name and make a copy in the aligned folder
    model1_name = allmodels[-1].split('/')[-1]
    model1.save('./outputs/{}/aligned/{}'.format(namethetest,model1_name))

    # --- Aligned each of the models to the model for the last time slice
    for model in allmodels[:-1]:

        model2_name = model.split('/')[-1]
        print('Now aligning {} to {}...'.format(model1_name,model2_name))
        
        ## -- Load model
        model2 = Word2Vec.load(model)

        ## -- Temp create a third model and save it in the aligned folder
        model3 = smart_procrustes_align_gensim(model1,model2)

        model3.save('./outputs/{}/aligned/{}'.format(namethetest,model2_name))

    print('Alignment complete!')
    print('Time to finish alignment: {} mins'.format(round((timeit.default_timer() - startalign) / 60, 2)))

print('Time to finish the whole process: {} mins'.format(round((timeit.default_timer() - startall) / 60, 2)))
