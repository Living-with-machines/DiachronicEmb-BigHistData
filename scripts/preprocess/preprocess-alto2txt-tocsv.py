# Adaptation of regexp.py to directly preprocess the alto2txt (instead of csvs)

import pandas as pd
import csv
import re
import nltk
from nltk.tokenize.regexp import regexp_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import timeit
from glob import glob

startall = timeit.default_timer()
collections = input('Which collections do you want to include? [lwm,hmd] ') or 'lwm,hmd'

cachedStopWords = stopwords.words("english")

def stopwdsrm(text):
    newtext = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return newtext

start = timeit.default_timer()
with open('../../../../datadrive/test-preprocess.csv', 'w') as outcsv:
    outpreprocessed = csv.writer(outcsv, delimiter=',')
    outpreprocessed.writerow(['NLP','date-of-publication','text'])

    for collection in collections.split(','):
        alltitles = glob('../../../../datadrive/plaintext-{}/*'.format(collection))

        for title in alltitles:
            nlptoprint = str(title.split('/')[6])
            allyears = glob('{}/*'.format(title))
            for year in allyears:
                yeartoprint = str(year.split('/')[7])
                allissues = glob('{}/*'.format(year))
                for issue in allissues:
                    textfiles = glob('{}/*.txt'.format(issue))
                    for textfile in textfiles:
                        print('Now processing {} ...'.format(textfile))
                        with open(textfile, 'r') as infile:
                            contents = infile.read()
                            text = re.sub('-\n', '', contents) # Remove OCR'd linebreaks within words
                            text = re.sub('\n', ' ', text) # Remove ordinary linebreaks
                            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text)) # Remove anything that is not a space, a letter, or a number
                            text = str(text).lower() # Lowercase
                            text = stopwdsrm(text) # Remove stopwords
                            outpreprocessed.writerow([nlptoprint,yeartoprint,text])
stop = timeit.default_timer()
print('Lowercased, punctuation and stopwords removed! It took {} minutes'.format((stop - start)/60))

# print('Tokenizing...')
# start = timeit.default_timer()
# newtext = []
# for s in tqdm(alltexts):
#     sentence = regexp_tokenize(s,pattern='\s+', gaps =True)
#     newtext.append(s)
# stop = timeit.default_timer()
# print('Tokenized! It took {} seconds'.format(stop - start))

stopall = timeit.default_timer()
print('Altogether it took {} seconds'.format(stopall - startall))

# with open('testprepro.txt','w') as testpre:
#     testpre.write('\n'.join(newtext))