# author: Antoine Tixier (see companion handout for a theoretical intro: http://www.lix.polytechnique.fr/~anti5662/intro_cnn_lstm_tixier.pdf)
# gets to 0.895 accuracy on the test set usually within 2 to 4 epochs (2 mins per epoch on NVidia TITAN)
# tested on ubuntu with Python 3, Keras version 1.1.0., tensorflow backend

import csv
import json
import numpy as np

from gensim import utils
from gensim.models.word2vec import Word2Vec

from keras.preprocessing import sequence
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Convolution1D, GlobalMaxPooling1D, Dense, Embedding, Input, Merge, Dropout

print('packages loaded')

path_to_IMDB = ''

# ========== read pre-processed data ==========

max_features = int(2e4)
stpwd_thd = 1 # TODO: allow greater values than 1 
max_size = int(4e2)
word_vector_dim = int(3e2)
do_non_static = True
nb_filters = 200
drop_rate = 0.3
batch_size = 32
nb_epoch = 10
my_optimizer = 'adam' # proved better than SGD and Adadelta

print('=== parameter values: ===')
print('top',max_features,'words used as features')
print('top',stpwd_thd-1,'words excluded')
print('max size of doc (in words):',max_size)
print('dim of word vectors:',word_vector_dim)
print('non-static:',do_non_static)
print('number of filters applied to each region:',nb_filters)
print('dropout rate:',drop_rate)
print('batch size:',batch_size)
print('number of epochs:',nb_epoch)
print('optimizer:',my_optimizer)

# ========== read pre-processed data ==========

# dictionary of word indexes (sorted by decreasing frequency across the corpus)
# this is a 1-based index - 0 is reserved for zero-padding
with open(path_to_IMDB + 'word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

with open(path_to_IMDB + 'training.csv', 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    x_train = list(reader)

with open(path_to_IMDB + 'test.csv', 'r') as my_file:
    reader = csv.reader(my_file, delimiter=',')
    x_test = list(reader)

with open(path_to_IMDB + 'training_labels.txt', 'r') as my_file:
    y_train = my_file.read().splitlines()

with open(path_to_IMDB + 'test_labels.txt', 'r') as my_file:
    y_test = my_file.read().splitlines()

# turn lists of strings into lists of integers
x_train = [[int(elt) for elt in sublist] for sublist in x_train]
x_test = [[int(elt) for elt in sublist] for sublist in x_test]  

y_train = [int(elt) for elt in y_train]
y_test = [int(elt) for elt in y_test]

print('data loaded')

# ========== pruning ==========

# only take into account the 'max_features' most frequent words
# disregard the 'stopword_threhsold' most frequent words

# to display the 'stopwords'
#[index_to_word[idx] for idx in range(1,stpwd_thd)]

x_train = [[elt for elt in rev if elt<=max_features and elt>=stpwd_thd] for rev in x_train]
x_test =  [[elt for elt in rev if elt<=max_features and elt>=stpwd_thd] for rev in x_test]

print('pruning done')

# ========== truncation and padding ==========

# truncate reviews of size larger than 'max_size' to their 'max_size' first words
x_train = [rev[:max_size] for rev in x_train]
x_test = [rev[:max_size] for rev in x_test]

# pad reviews shorter than 'max_size'
# the vector of the 0th index will be set to all zeroes (zero padding strategy)

x_train = [rev+[0]*(max_size-len(rev)) if len(rev)<max_size else rev for rev in x_train]

# sanity check: all reviews should now be of size 'max_size'
if max_size == list(set([len(rev) for rev in x_train]))[0]:
    print('1st sanity check passed')
else:
    print('1st sanity check failed !')

x_test = [rev+[0]*(max_size-len(rev)) if len(rev)<max_size else rev for rev in x_test]

if max_size == list(set([len(rev) for rev in x_test]))[0]:
    print('2nd sanity check passed')
else:
    print('2nd sanity check failed !')

print('truncation and padding done')

# ========== loading pre-trained word vectors ==========

# invert mapping
index_to_word = dict((v,k) for k, v in word_to_index.items())

# convert integer reviews into word reviews
x_full = x_train + x_test

x_full_words = [[index_to_word[idx] for idx in rev if idx!=0] for rev in x_full]

# initialize word vectors
word_vectors = Word2Vec(size=word_vector_dim, min_count=1)

# create entries for the words in our vocabulary
word_vectors.build_vocab(x_full_words)

# sanity check
all_words = [word for rev in x_full_words for word in rev]

if len(list(set(all_words))) == len(word_vectors.wv.vocab):
    print('3rd sanity check passed')
else:
    print('3rd sanity check failed !')

# fill entries with the pre-trained word vectors
path_to_pretrained_wv = ''
word_vectors.intersect_word2vec_format(path_to_pretrained_wv + 'GoogleNews-vectors-negative300.bin', binary=True)

print('pre-trained word vectors loaded')

# NOTE: in-vocab words without an entry in the binary file are not removed from the vocabulary
# instead, their vectors are silently initialized to random values

# if necessary, we can detect those vectors via their norms which approach zero
#norms = [np.linalg.norm(word_vectors[word]) for word in word_vectors.wv.vocab.keys()]
#idxs_zero_norms = [idx for idx, norm in enumerate(norms) if norm<0.05]
# most of those words are proper nouns, like Patton, Deneuve, etc.
# they don't have an entry in the word vectors because we lowercased the text
#no_entry_words = [word_vectors.wv.vocab.keys()[idx] for idx in idxs_zero_norms]

# create numpy array of embeddings
embeddings = np.zeros((max_features + 1,word_vector_dim))
for word in word_vectors.wv.vocab.keys():
    idx = word_to_index[word]
    # word_to_index is 1-based! the 0-th row, used for padding, stays at zero
    embeddings[idx,] = word_vectors[word]

print('embeddings created')

# ========== training CNN ==========

my_input = Input(shape=(max_size,), dtype='int32') # for some reason here it is important to let the second argument of shape blank

embedding = Embedding(input_dim = len(word_vectors.wv.vocab) + 1, # vocab size, including the 0-th word used for padding
                      output_dim = word_vector_dim,
                      #input_length = max_size, # length of input sequences
                      dropout = drop_rate, # see http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
                      #embeddings_constraint = maxnorm(3.),
                      weights=[embeddings], # we pass our pre-trained embeddings
                      trainable = do_non_static # non-static mode
                      ) (my_input)

# size of the feature map should be equal to max_size-filter_length+1
conv_1 = Convolution1D(nb_filter = nb_filters,
                       filter_length = 4, # region size
                       activation='relu',
                       ) (embedding)

pooled_conv_1 = GlobalMaxPooling1D() (conv_1)

conv_2 = Convolution1D(nb_filter = nb_filters,
                       filter_length = 5, # region size
                       activation='relu',
                       ) (embedding)

pooled_conv_2 = GlobalMaxPooling1D() (conv_2)

conv_3 = Convolution1D(nb_filter = nb_filters,
                       filter_length = 6, # region size
                       activation='relu',
                       ) (embedding)

pooled_conv_3 = GlobalMaxPooling1D() (conv_3)

merge = Merge(mode='concat') ([pooled_conv_1,pooled_conv_2,pooled_conv_3])

merge_dropped = Dropout(drop_rate) (merge) # adding this layer improved test set accuracy by almost 2%

# we finally project onto a single unit output layer with sigmoi activation
prob = Dense(output_dim = 1, # dimensionality of the output space
             activation='sigmoid'#,
             #W_constraint = maxnorm(3.) # constrain L-2 norm of the weights. Slows up convergence (more epochs needed), but does not improve performance. In most recent version of Keras this argument has been renamed 'kernel_constraint'
             ) (merge_dropped)

model = Model(my_input, prob)

model.compile(loss='binary_crossentropy',
              optimizer=my_optimizer,
              metrics=['accuracy'])

print('model compiled')

early_stopping = EarlyStopping(monitor='val_acc', # we go through the epochs as long as the accuracy on the validation set increases
                               patience = 3,
                               mode = 'max')

model.fit(x_train, 
          y_train,
          batch_size = batch_size,
          nb_epoch = nb_epoch,
          validation_data = (x_test, y_test),
          callbacks = [early_stopping])

# persist model to disk

model.save(path_to_IMDB + 'imdb_cnn_3_30_17')

print('model saved to disk')

#loss, acc = model.evaluate(x_test, y_test, batch_size = batch_size)

#print('final accuracy on test set:', acc)
