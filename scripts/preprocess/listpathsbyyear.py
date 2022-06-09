import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.regexp import regexp_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import timeit
from glob import glob
import os


timespan = input('Enter the timespan to consider in the format YYYY-YYYY (default: 1780-1920): ') or '1780-1920'
start_year = float(timespan.split('-')[0]) # Explicitly set the start year
end_year = float(timespan.split('-')[1]) # Explicitly set the end year
period_length = input('Press Enter to divide the timespan by decades or enter the interval you wish (e.g. 20 for 20-year periods): ') or 10 # Set a 10-year increment (i.e. train one w2v model per decade)

collections = input('Which collections do you want to include? [lwm,hmd] ') or 'lwm,hmd'

year_range = end_year - start_year # Year range, e.g. 1890-1830 = 60
modulo = year_range % period_length # How many possible periods there will be, e.g. 60 % 10 = 6
if modulo == 0: # In case there's only 1 year represented
    final_start = end_year - period_length
else:
    final_start = end_year - modulo # This is a slightly complicated way of saying: set a numeber just below your end year.
final_end = end_year + 1
starts = np.arange(start_year, final_start, period_length).tolist()
print('Starts: ')
print(str(starts))
tuples = [(start, start+period_length) for start in starts]
tuples.append(tuple([final_start, final_end]))
print('Tuples: ')
print(str(tuples))
bins = pd.IntervalIndex.from_tuples(tuples, closed='left')
print('Bins: ')
print(str(bins))
original_labels = list(bins.astype(str))
print('Original_labels: ')
print(str(original_labels))
new_labels = ['{} - {}'.format(b.strip('[)').split(', ')[0], float(b.strip('[)').split(', ')[1])-1) for b in original_labels]
print('New labels: ')
print(str(new_labels))
label_dict = {original_labels[i]: new_labels[i] for i in range(len(original_labels))}
print('Label dict: ')
print(str(label_dict))

allpaths = []
for collection in collections.split(','):
    alltitles = glob('../../../../datadrive/plaintext-{}/*'.format(collection))
    for title in alltitles:
        allyears = glob('{}/*'.format(title))
        allpaths = allpaths + allyears

p = {'paths': allpaths}
paths = pd.DataFrame(data=p)

paths['year'] = [int(str(x).split('/')[7]) for x in paths['paths']]
print(paths['year'])
paths['period'] = pd.cut(paths['year'], bins=bins, include_lowest=True, precision=0)
paths['period'] = paths['period'].astype("str")
df = paths.replace(label_dict)

df.to_csv('paths_by_period.csv',index=False)