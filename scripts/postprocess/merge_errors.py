# Given a model for one decade it applies pyspellchecker to the whole vocabulary, merging vectors that are
# considered to be mispellings with the 'correct' word vector by summing and averaging the vector

from gensim.models import Word2Vec
from spellchecker import SpellChecker
from tqdm import tqdm
from collections import defaultdict
import os

spell = SpellChecker()

samplename = input('Enter name of sample: ') or 'lwmhmdwhole'
decade = input('Enter one decade: ') or '1860'

model = Word2Vec.load('./outputs/{}/aligned/{}s.model'.format(samplename,decade))

model.wv.save_word2vec_format('./outputs/{}/aligned/{}s-vectors.txt'.format(samplename,decade), binary=False) # Also save the vectors only (easier to work with) - Not necessary, of course

if not os.path.exists('./outputs/{}/postcorrected'.format(samplename)):
    os.mkdir('./outputs/{}/postcorrected'.format(samplename))

allwords = defaultdict(dict)
with open('./outputs/{}/aligned/{}s-vectors.txt'.format(samplename,decade),'r') as allvectors:
    for line in tqdm(allvectors):
        word = line.split(' ')[0]
        item2 = spell.correction(word)
        if item2 != word:
            allwords[word] = item2

# pretrained_vectors = KeyedVectors.load_word2vec_format('./outputs/{}/aligned/{}s-vectors.txt'.format(samplename,decade), binary=False)
# print(pretrained_vectors['machine'])

for k,v in tqdm(allwords.items()):
    if k != v:
        try:
            model.wv[v] = model.wv[v] + model.wv[k]
        except KeyError:
            continue

# # model.wv.save_word2vec_format('./outputs/{}/aligned/{}s-vectors.txt'.format(samplename,decade), binary=False) # Also save the vectors only (easier to work with) - Not necessary, of course

outvectors = open('./outputs/{}/postcorrected/{}s-vectors-corrected.txt'.format(samplename,decade),'w')
with open('./outputs/{}/aligned/{}s-vectors.txt'.format(samplename,decade),'r') as allvectors:
    for line in allvectors:
        word = line.split(' ')[0]
        if word not in allwords.keys():
            outvectors.write(line)


with open('./outputs/{}/postcorrected/{}s-vectors-corrected.txt'.format(samplename,decade), 'r') as intxt:
    lines = intxt.readlines()
    count = len(lines) - 1

veclen = lines[0].split('\n')[0].split(' ')[1]
lines[0] = "{} {}\n".format(str(count),veclen)

with open('./outputs/{}/postcorrected/{}s-vectors-corrected.txt'.format(samplename,decade),'w') as f:
    f.writelines(lines)

# # # print(len(allwords))


# # # with open('misspellstocorrected.csv', 'w') as csvwriter:
# # #     csvfile = csv.writer(csvwriter)
# # #     for item in tqdm(allwords):
# # #         item2 = spell.correction(item)
# # #         csvfile.writerow([item, item2])