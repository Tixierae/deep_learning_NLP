import json
import operator
import numpy as np
from collections import Counter
from nltk import word_tokenize

# = = = = = = = = = = = = = = = = = = = = =

def get_vocab(source_or_target,pairs):
    '''pairs is of the form [[source_doc_1,target_doc_1],...,[source_doc_n,target_doc_n]]'''
    if source_or_target == 'source':
        my_idx = 0
    elif source_or_target == 'target':
        my_idx = 1
    
    all_docs = [elt[my_idx] for elt in pairs]
    all_tokens = [elt for sublist in all_docs for elt in sublist] # flatten
    counts = dict(Counter(all_tokens)) # get token frequencies
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), \
                           reverse=True) # assign to each word an index based on its frequency
    word_to_idx = dict([(my_t[0],idx) for idx,my_t in enumerate(sorted_counts,idx_mf) \
                        if my_t[1]>=min_count]) # works because 'sorted_counts' is sorted in decreasing order!
    return word_to_idx

def convert_to_ints(pairs):
    pairs_int = []
    for pair in pairs:
        source = [str(word_to_idx_source[elt]) if elt in word_to_idx_source else \
                  oov_token for elt in pair[0]] #+ [eos_token]
        target = [str(word_to_idx_target[elt]) if elt in word_to_idx_target else \
                  oov_token for elt in pair[1]]
        target = target #+ [eos_token]
        pairs_int.append([source,target])
    
    return pairs_int

# = = = = = = = = = = = = = = = = = = = = =

path_to_data = './data/'

min_count = 5
padding_token = '0'
oov_token = '1'
sos_token = '2' # start of sentence, only needed for the target sentence
eos_token = '3' # end of sentence, only needed for the target sentence
idx_mf = 4

# = = = = = = = = = = = = = = = = = = = = =

with open(path_to_data + 'eng_fr.txt', 'r', encoding='utf-8') as file:
    pairs = file.read().splitlines()
    
pairs = [elt.lower().replace("'",' ').replace('-',' ').split('\t')[:2] for \
         elt in pairs] # apostrophes and dashes are removed to lower the nb of unique words

pairs = [[word_tokenize(elt[0],'english'),word_tokenize(elt[1],'french')] for elt in pairs]

source_lens = [len(elt[0]) for elt in pairs]
target_lens = [len(elt[1]) for elt in pairs]

print('source sentence size: min: %s, max: %s, mean: %s' % (min(source_lens),max(source_lens),np.mean(source_lens)))
print('target sentence size: min: %s, max: %s, mean: %s'% (min(target_lens),max(target_lens),np.mean(target_lens)))

# split train/test
test_idxs = np.random.choice(range(len(pairs)),size=int(0.2*len(pairs)),replace=False)
train_idxs = list(set(range(len(pairs))).difference(set(test_idxs)))
assert len(test_idxs) + len(train_idxs) == len(pairs)

pairs_train = [pairs[idx] for idx in train_idxs]
pairs_test = [pairs[idx] for idx in test_idxs]

# create dicts from the training set
word_to_idx_source = get_vocab('source',pairs_train)
word_to_idx_target = get_vocab('target',pairs_train)

with open(path_to_data + 'vocab_source.json', 'w') as file:
    json.dump(word_to_idx_source, file, sort_keys=True, indent=4)

with open(path_to_data + 'vocab_target.json', 'w') as file:
    json.dump(word_to_idx_target, file, sort_keys=True, indent=4)

# transform into indexes
pairs_train_ints = convert_to_ints(pairs_train)
pairs_test_ints = convert_to_ints(pairs_test)

# save to disk
with open(path_to_data + 'pairs_train_ints.txt', 'w') as file:
    for elt in pairs_train_ints:
        to_write = ' '.join(elt[0]) + '\t' + ' '.join(elt[1]) + '\n'
        file.write(to_write)

with open(path_to_data + 'pairs_test_ints.txt', 'w') as file:
    for elt in pairs_test_ints:
        to_write = ' '.join(elt[0]) + '\t' + ' '.join(elt[1]) + '\n'
        file.write(to_write)

