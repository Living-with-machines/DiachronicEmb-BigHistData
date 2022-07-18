from gensim.models import Word2Vec

samplename1 = input('Enter name of sample1: ')
samplename2 = input('Enter name of sample2: ')
decade = input('Enter one decade to compare across the two models: ')

model_one = Word2Vec.load('./outputs/{}/raw/{}s.model'.format(samplename1,decade))
model_two = Word2Vec.load('./outputs/{}/raw/{}s.model'.format(samplename2,decade))

print('Most similar to machine: ')
print(model_one.wv.most_similar('machine', topn=10))
print(model_two.wv.most_similar('machine', topn=10))

print('Most similar to machines: ')
print(model_one.wv.most_similar('machines', topn=10))
print(model_two.wv.most_similar('machines', topn=10))

print('Most similar to machinery: ')
print(model_one.wv.most_similar('machinery', topn=10))
print(model_two.wv.most_similar('machinery', topn=10))