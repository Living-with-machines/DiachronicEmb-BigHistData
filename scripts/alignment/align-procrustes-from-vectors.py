# Aligns models if only the vectors are available (not recommended unless you don't have the full model anymore)

import numpy as np
from gensim.models import KeyedVectors
from glob import glob
import os

samplename = input('Enter sample name (name of folder under outputs/ with one w2v model per decade) (e.g. lwmhmdwhole): ')

if not os.path.exists('./outputs/{}/aligned'.format(samplename)):
    os.mkdir('./outputs/{}/aligned'.format(samplename))

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.

    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # re-filling the normed vectors
    in_base_embed.fill_norms(force=True)
    in_other_embed.fill_norms(force=True)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.get_normed_vectors()
    other_vecs = in_other_embed.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.vectors = (other_embed.vectors).dot(ortho)    
    
    return other_embed

def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.index_to_key)
    vocab_m2 = set(m2.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.get_vecattr(w, "count") + m2.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.key_to_index[w] for w in common_vocab]
        old_arr = m.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.key_to_index = new_key_to_index
        m.index_to_key = new_index_to_key
        
        print(len(m.key_to_index), len(m.vectors))
        
    return (m1,m2)

allmodels = glob('./outputs/{}/raw/*vectors.txt'.format(samplename))
allmodels = sorted(allmodels)
print(allmodels)
firstmodel = KeyedVectors.load_word2vec_format(allmodels[0],binary=False)
modelname = allmodels[0].split('/')[-1]
firstmodel.save('./outputs/{}/aligned/{}'.format(samplename,modelname))

for model in allmodels:
    if model != allmodels[-1]:
        print('Now aligning {} to {}...'.format(model,str(allmodels[allmodels.index(model) + 1])))
        model1 = KeyedVectors.load_word2vec_format(model)
        indexmodel1 = allmodels.index(model)
        modelname = allmodels[indexmodel1 + 1].split('/')[-1]
        model2 = KeyedVectors.load_word2vec_format(allmodels[indexmodel1 + 1],binary=False)

        model3 = smart_procrustes_align_gensim(model1,model2)
        with open('aligned-test.txt','w') as outtxt:
            for key,value in model3.key_to_index.items():
                outtxt.write(str(key + ' ') + str(' '.join(map(str, model3[key].tolist()))) + '\n')
    else:
        print('Alignment complete!')