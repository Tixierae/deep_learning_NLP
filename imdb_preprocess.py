# written by Antoine Tixier to go with the following handout:
# data initially taken from: http://ai.stanford.edu/~amaas/data/sentiment/

import os
import re
import random
import operator
import csv
import json
from nltk import pos_tag
from collections import Counter
from bs4 import BeautifulSoup

path_root = 'C:\\Users\\mvazirg\\'

path_to_IMDB = path_root + 'Desktop\\IMDB\\aclImdb\\'

# ========== read pos/neg reviews from the training and test sets ==========

train_test = ['train','test']
neg_pos = ['neg','pos']

reviews = []
labels = []

for elt in train_test:
    print '==== going over', elt, 'set ===='
    my_path = path_to_IMDB + elt + '\\'
    for p, pol in enumerate(neg_pos):
        review_names = os.listdir(my_path + pol + '\\')
        print len(review_names), pol, 'reviews found'
        counter = 0
        for review_name in review_names:
            # read .txt file
            with open(my_path + pol + '\\' + review_name, 'r') as my_file: 
                reviews.append(my_file.read())
                labels.append(p)
                if counter % 1e3 == 0:
                    print counter, '/', len(review_names), 'reviews read'
                counter += 1

print len(reviews), 'reviews and', len(labels), 'labels assigned'

# ========== clean reviews ==========

# regex to match intra-word apostrophes
regex_ap = re.compile(r"(\b[']\b)|[\W_]")

cleaned_reviews = []
counter = 0

for rev in reviews:
    # remove HTML formatting
    temp = BeautifulSoup(rev)
    text = temp.get_text()
    text = text.lower()
    # replace punctuation with whitespace
    # note that we exclude apostrophes from our list of punctuation marks: we want to keep don't, shouldn't etc.
    text = re.sub(r'[()\[\]{}.,;:!?\<=>?@^_`~#$%"&*-]', ' ', text)
    # remove apostrophes that are not intra-word
    text = regex_ap.sub(lambda x: (x.group(1) if x.group(1) else ' '), text)
    # strip extra white space
    text = re.sub(' +',' ',text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize
    tokens = text.split()
    # remove single letter tokens (we don't remove stopwords as some of them might be useful in determining polarity, like not, but...)
    tokens = [tok for tok in tokens if len(tok)>1]
    # POS tag
    #tagged_tokens = pos_tag(tokens)
    # convert to lower case words that are not identified as proper nouns
    #tokens = [token.lower() if tagged_tokens[idx][1]!='NNP' else token for idx,token in enumerate(tokens)]
    # save
    cleaned_reviews.append(tokens)
    if counter % 1e3 == 0:
        print counter, '/', len(reviews), 'reviews cleaned'
    counter += 1

# get list of tokens from all reviews
all_tokens = [token for sublist in cleaned_reviews for token in sublist]

counts = dict(Counter(all_tokens))

sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

# assign to each word an index based on its frequency in the corpus
# the most frequent word will get index equal to 1

word_to_index = dict([(tuple[0],idx+1) for idx, tuple in enumerate(sorted_counts)])

# save dictionary
with open(path_to_IMDB + 'word_to_index.json', 'w') as my_file:
    json.dump(word_to_index, my_file, sort_keys=True, indent=4)

# e.g., 'the' is the most frequent word, "don't" is the 86th most recent word
word_to_index['the']
word_to_index["don't"]

# transform each review into a list of word indexes
cleaned_reviews_integers = []
counter = 0

for rev in cleaned_reviews:
    sublist = []
    for token in rev:
        sublist.append(word_to_index[token])
    cleaned_reviews_integers.append(sublist)
    if counter % 1e4 == 0:
        print counter, '/', len(reviews), 'reviews cleaned'
    counter += 1

# ========== split training/testing, shuffle, and save to disk ==========

idx_split = 25000
training_rev = cleaned_reviews_integers[:idx_split]
training_labels = labels[:idx_split]

test_rev = cleaned_reviews_integers[idx_split:]
test_labels = labels[idx_split:]

random.seed(3272017)
shuffle_train = random.sample(range(len(training_rev)), len(training_rev))
shuffle_test = random.sample(range(len(test_rev)), len(test_rev))

training_rev = [training_rev[shuffle_train[elt]] for elt in shuffle_train]
training_labels = [training_labels[shuffle_train[elt]] for elt in shuffle_train]

test_rev = [test_rev[shuffle_test[elt]] for elt in shuffle_test]
test_labels = [test_labels[shuffle_test[elt]] for elt in shuffle_test]

# 'wb' instead of 'w' ensures that the rows are not separated with blank lines
with open(path_to_IMDB + 'training.csv', 'wb') as my_file:
    writer = csv.writer(my_file, quoting=csv.QUOTE_NONE)
    writer.writerows(training_rev)

with open(path_to_IMDB + 'training_labels.txt', 'wb') as my_file:
    for label in training_labels:
        my_file.write(str(label) + '\n')
 
with open(path_to_IMDB + 'test.csv', 'wb') as my_file:
    writer = csv.writer(my_file, quoting=csv.QUOTE_NONE)
    writer.writerows(test_rev)

with open(path_to_IMDB + 'test_labels.txt', 'wb') as my_file:
    for label in test_labels:
        my_file.write(str(label) + '\n')

print 'all results saved to disk'