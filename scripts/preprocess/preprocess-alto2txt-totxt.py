# Adaptation of regexp.py to directly preprocess the alto2txt (instead of csvs)

import pandas as pd
# import csv
import re
import nltk
# from nltk.tokenize.regexp import regexp_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import timeit
from glob import glob
import os


startall = timeit.default_timer()

namethetest = input('Give a name to this test run: ')

pathsperiods = pd.read_csv('paths_by_period.csv')
pathsperiods = [x for _, x in pathsperiods.groupby('period')]

if not os.path.exists('../../../../datadrive/{}'.format(namethetest)):
    os.mkdir('../../../../datadrive/{}'.format(namethetest))

cachedStopWords = stopwords.words("english")

def stopwdsrm(text):
    newtext = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return newtext

for df in pathsperiods:
    start = timeit.default_timer()
    decade = str(df['period'].iloc[0]).split(' - ')[0].split('.')[0] # Get just the first part of the range to represent the decade (e.g. '1840.0 - 1849.0' --> 1840s)
    newfile = open('../../../../datadrive/{}/{}s.txt'.format(namethetest,decade), 'w') # Create 1 txt file per decade
    for path in df['paths']:
        allissues = glob('{}/*'.format(path))
        for issue in allissues:
            textfiles = glob('{}/*.txt'.format(issue))
            for textfile in textfiles:
                print('Now processing {} ...'.format(textfile))
                with open(textfile, 'r') as infile:
                    contents = infile.read()
                    text = re.sub('-\n', '', contents) # Remove OCR'd linebreaks within words
                    text = re.sub('\n', ' ', text) # Remove ordinary linebreaks
                    sentences = sent_tokenize(text) # Split into sentences
                    for sentence in sentences:
                        sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(sentence)) # Remove anything that is not a space, a letter, or a number
                        sentence = str(sentence).lower() # Lowercase
                        sentence = stopwdsrm(sentence) # Remove stopwords
                        if sentence != ' ':
                            newfile.write(sentence + '\n') # Write sentence as a new line in the txt file
    stop = timeit.default_timer()
    print('{} Done! Lowercased, punctuation and stopwords removed! It took {} minutes'.format(decade, (stop - start)/60))

# print('Tokenizing...')
# start = timeit.default_timer()
# newtext = []
# for s in tqdm(alltexts):
#     sentence = regexp_tokenize(s,pattern='\s+', gaps =True)
#     newtext.append(s)
# stop = timeit.default_timer()
# print('Tokenized! It took {} seconds'.format(stop - start))
stopall = timeit.default_timer()
print('All preprocessing finished. Altogether it took {} minutes'.format((stopall - startall) / 60))

# with open('testprepro.txt','w') as testpre:
#     testpre.write('\n'.join(newtext))