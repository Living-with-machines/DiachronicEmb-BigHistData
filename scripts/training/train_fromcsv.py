import numpy as np
import pandas as pd
from nltk.tokenize.regexp import regexp_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from tqdm import tqdm
from gensim.models import Phrases
import timeit


# PREPARE THE DF

timespan = input('Enter the timespan to consider in the format YYYY-YYYY (default: 1780-1920): ') or '1780-1920'
start_year = float(timespan.split('-')[0]) # Explicitly set the start year
end_year = float(timespan.split('-')[1]) # Explicitly set the end year
period_length = input('Press Enter to divide the timespan by decades or enter the interval you wish (e.g. 20 for 20-year periods): ') or 10 # Set a 10-year increment (i.e. train one w2v model per decade)

bigrams = input('Do you want to train a collocation (bigram detector) (y or [n])? ') or 'n'

startall = timeit.default_timer()

dfloadtime = timeit.default_timer()
df = pd.read_csv('/Users/npedrazzini/Desktop/LwMWordEmb/input_data/lwm-whole-preprocessed.csv', usecols=['date-of-publication','text'])
print('Time to load the df: {} mins'.format(round((timeit.default_timer() - dfloadtime) / 60, 2)))

# SPLIT DF BY PERIOD
year_range = end_year - start_year # Year range, e.g. 1890-1830 = 60
modulo = year_range % period_length # How many possible periods there will be, e.g. 60 % 10 = 6
if modulo == 0: # In case there's only 1 year represented
    final_start = end_year - period_length
else:
    final_start = end_year - modulo # This is a slightly complicated way of saying: set a numeber just below your end year.
final_end = end_year + 1
starts = np.arange(start_year, final_start, period_length).tolist()
tuples = [(start, start+period_length) for start in starts]
tuples.append(tuple([final_start, final_end]))
bins = pd.IntervalIndex.from_tuples(tuples, closed='left')
original_labels = list(bins.astype(str))
new_labels = ['{} - {}'.format(b.strip('[)').split(', ')[0], float(b.strip('[)').split(', ')[1])-1) for b in original_labels]
label_dict = {original_labels[i]: new_labels[i] for i in range(len(original_labels))}

# ADD COLUMN 'PERIOD' TO DF
df['period'] = pd.cut(df['date-of-publication'], bins=bins, include_lowest=True, precision=0)
df['period'] = df['period'].astype("str")
df = df.replace(label_dict)

# CREATE LIST OF DFs GROUPED BY COLUMN 'PERIOD'
df = [x for _, x in df.groupby('period')]

# WORD2VEC
# TRAIN ONE MODEL PER DF
for dff in df:
	w2v_model = Word2Vec(min_count=10,
                    window=5, 
                    sg=True,
                    vector_size=200)

	decadetime = timeit.default_timer()

	decade = str(dff['period'].iloc[0]).split(' - ')[0].split('.')[0] # Get the decade as, e.g., '1870' if it is 1870-1880. We'll use this string to name the model
	print('Now working on the {}s'.format(decade))
	sentences = [] # Here we'll store all the tokenized sentences
	texts = dff['text']
	print(texts)
	print('Tokenizing...')
	starttokenize = timeit.default_timer()
	for text in tqdm(texts):
		try:
			sentence = regexp_tokenize(text,pattern='\s+', gaps =True)
			print(sentence)
			sentences.append(sentence)
		except TypeError:
			continue
	print('Tokenized! It took {} seconds'.format(timeit.default_timer() - starttokenize))

    # TRAIN A BIGRAM DETECTOR TO ALLOW CAPTURING TWO-WORD COLLOCATIONS (E.G. SEWING MACHINE) 
    
	if bigrams == 'y':
		print('Applying bigram transformation...')
		sentences = Phrases(sentences)
		print('Building vocabulary...')
	try:
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
                        epochs=10)
	else:
		w2v_model.train(sentences, 
                    total_examples=w2v_model.corpus_count, 
                    start_alpha=0.025, 
                    end_alpha=0.0025, 
                    epochs=10)
    
	w2v_model.save("/Users/npedrazzini/Desktop/LwMWordEmb/outputs/lwmcollection/{}s-word2vec-newspapers.model".format(decade))

	print('Time to train the model: {} mins'.format(round((timeit.default_timer() - traintime) / 60, 2)))
    
	print('Time to finish a decade: {} mins'.format(round((timeit.default_timer() - decadetime) / 60, 2)))

print('Time to finish the whole process: {} mins'.format(round((timeit.default_timer() - startall) / 60, 2)))