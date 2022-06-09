from random import sample
import numpy as np
import pandas as pd
from nltk.tokenize.regexp import regexp_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from tqdm import tqdm
from gensim.models import Phrases
import timeit
from glob import glob
import os


# PREPARE THE DF

samplename = input('Enter name of sample (i.e. name of folder under inputs/ containing the plain txt files): ')
timespan = input('Enter the timespan to consider in the format YYYY-YYYY (default: 1780-1920): ') or '1780-1920'
start_year = float(timespan.split('-')[0]) # Explicitly set the start year
end_year = float(timespan.split('-')[1]) # Explicitly set the end year
period_length = input('Press Enter to divide the timespan by decades or enter the interval you wish (e.g. 20 for 20-year periods): ') or 10 # Set a 10-year increment (i.e. train one w2v model per decade)

bigrams = input('Do you want to train a collocation (bigram detector) (y or [n])? ') or 'n'
# spellcheck = ('Do you want to apply a spellchecker? (y or [n])? ') or 'n'

startall = timeit.default_timer()

allfiles = glob('./input_data/{}/*'.format(samplename))

if not os.path.exists('./outputs/{}'.format(samplename)):
	os.mkdir('./outputs/{}'.format(samplename))

# WORD2VEC
# TRAIN ONE MODEL PER DF
for file in allfiles:
	decadetime = timeit.default_timer()
	decade = file.split('/')[-1].split('s.')[0]
	print('Now working on the {}s'.format(decade))
	sentences = []
	with open(file, 'r') as infile:
		starttokenize = timeit.default_timer()
		print('Now tokenizing...')
		for line in tqdm(infile.readlines()):
			if line != '':
				sentence = regexp_tokenize(line,pattern='\s+', gaps =True)
				sentences.append(sentence)
		print('Tokenized! It took {} seconds'.format(timeit.default_timer() - starttokenize))


	w2v_model = Word2Vec(min_count=10,
                    window=5, 
                    sg=True,
                    vector_size=200)

    # TRAIN A BIGRAM DETECTOR TO ALLOW CAPTURING TWO-WORD COLLOCATIONS (E.G. SEWING MACHINE) 
	if bigrams == 'y':
		print('Applying bigram transformation...')
		sentences = Phrases(sentences)
		print('Building vocabulary...')
	try:
		print('Building vocabulary...')
		w2v_model.build_vocab(sentences, progress_per=10000, update=True)
	except RuntimeError:
		w2v_model.build_vocab(sentences, progress_per=10000)

	print('Now training the {}s model... This might take a very long time. Have a coffee. Or two.'.format(decade))
	traintime = timeit.default_timer()
	if bigrams == 'y':
		w2v_model.train(sentences[sentences],
                        total_examples=w2v_model.corpus_count, 
                        start_alpha=0.025, 
                        end_alpha=0.0025, 
                        epochs=5)
	else:
		w2v_model.train(sentences, 
                    total_examples=w2v_model.corpus_count, 
                    start_alpha=0.025, 
                    end_alpha=0.0025, 
                    epochs=5)
    
	w2v_model.save("./outputs/{}/{}s.model".format(samplename,decade))

	print('Time to train the model: {} mins'.format(round((timeit.default_timer() - traintime) / 60, 2)))
    
	print('Time to finish the decade: {} mins'.format(round((timeit.default_timer() - decadetime) / 60, 2)))

print('Time to finish the whole process: {} mins'.format(round((timeit.default_timer() - startall) / 60, 2)))