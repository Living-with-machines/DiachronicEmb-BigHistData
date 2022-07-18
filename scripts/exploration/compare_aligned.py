from scipy.spatial.distance import cosine

from gensim.models import Word2Vec

samplename = input('Enter name of sample: ')
decade1 = input('Enter one decade: ')
decade2 = input('Enter another decade: ')
word = input('Enter a word you want to check (lowercase): ')

model_one = Word2Vec.load('./outputs/{}/raw/{}s.model'.format(samplename,decade1))
model_two = Word2Vec.load('./outputs/{}/raw/{}s.model'.format(samplename,decade2))

machine_sim1 = model_one.wv.most_similar(word, topn=100)
machine_sim2 = model_two.wv.most_similar(word, topn=100)

print('Non-aligned, nearest neighbours for {} in {}: '.format(word,decade1), str(machine_sim1))
print('Non-aligned, nearest neighbours for {} in {}: '.format(word,decade2), str(machine_sim2))

print('Cosine similarities between {} in {} and {}, non-aligned: '.format(word,decade1,decade2), (1-cosine(model_one.wv[word], model_two.wv[word])))


# Now the aligned
model_one = Word2Vec.load('./outputs/{}/aligned/{}s.model'.format(samplename,decade1))
model_two = Word2Vec.load('./outputs/{}/aligned/{}s.model'.format(samplename,decade2))

machine_sim1 = model_one.wv.most_similar(word, topn=100)
machine_sim2 = model_two.wv.most_similar(word, topn=100)

print('Aligned, nearest neighbours for {} in {}: '.format(word,decade1), str(machine_sim1))
print('Aligned, nearest neighbours for {} in {}: '.format(word,decade2), str(machine_sim2))

print('Cosine similarities between {} in {} and {}, aligned: '.format(word,decade1,decade2), (1-cosine(model_one.wv[word], model_two.wv[word])))
