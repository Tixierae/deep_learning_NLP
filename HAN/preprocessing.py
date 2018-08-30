'''
TODO update description!
this script implements the following preprocessing steps:
    - split training set into training (90%) and validation (10%)
    - tokenize docs into sentences and sentences into words
    - count the number of occurrences of each word
    - replace the unfrequent words with a special token
    - turn word docs into integer docs, sorted by increasing # of sents (for bucketing)
    - learn word2vec embeddings from training set (devoid of validation set)

line by line processing is used, so datasets that don't fit in RAM can still be treated

Example of one line of the input files (Amazon review dataset format):

"1","mens ultrasheer","This model may be ok for sedentary types, but I'm active 
and get around alot in my job - consistently found these stockings rolled up down 
by my ankles! Not Good!! Solution: go with the standard compression stocking, 
20-30, stock #114622. Excellent support, stays up and gives me what I need. 
Both pair of these also tore as I struggled to pull them up all the time. 
Good riddance/bad investment!"

warnings: 
    - input files should be named 'train.csv' and 'test.csv', be in format above
    - last line of each file should be blank!
    - all available cores are used by default (hardcode 'n_jobs' to change)
    - all generated files should be deleted before re-running the script in the same dir

tested on Python 3.5 and 3.6 with gensim 3.2.0
takes ~2.17 hours to process full Amazon dataset (3.65M docs, 1.6GB) using 8 cores @2.4GHz

'''

import os
import re
import json
import nltk
import math
import random
import operator
import multiprocessing
import numpy as np

from time import time
from random import shuffle
from functools import partial
from collections import Counter
from multiprocessing import Pool
from nltk import sent_tokenize, word_tokenize
from gensim.models.word2vec import LineSentence, Word2Vec, FAST_VERSION 

# draft commands below about using Stanford CoreNLP tokenizer instead of NLTK
# from subprocess import call, Popen
# path_to_corenlp = 'H:\\stanford-corenlp-full-2018-02-27\\'
# my_command = 'java -mx300m -classpath "*" edu.stanford.nlp.process.PTBTokenizer' + path_to_tmp + 'sample.txt ' + '> ' + path_to_tmp + 'sample_tok.txt'
# os.chdir(path_to_corenlp)
# my_command = 'java -mx300m -classpath "*" edu.stanford.nlp.process.DocumentPreprocessor doc.txt > sents.txt -tokenizePerLine=true -tokenizeNLs=false'
# Popen(my_command, shell=True, cwd=path_to_corenlp)
# my_command = 'java -mx300m -classpath "*" edu.stanford.nlp.process.PTBTokenizer sents.txt > tokens.txt -preserveLines'
# Popen(my_command, shell=True, cwd=path_to_corenlp)

# ============================== PATHS & PARAMETERS ==============================

dataset_name = 'amazon_review_full_csv' 
one_based_labels = True # Remember to change this when changing dataset!E.g., should be False for IMDB.  

is_GPU = True # just a flag to select the right paths for the machine

if is_GPU:
    path_to_data = '/home/antoine/Desktop/TextClassificationDatasets/' + dataset_name + '/'
    path_to_batches = path_to_data + 'batches/'
else:
    path_to_data = 'H:\\' + dataset_name +'\\'
    path_to_batches = path_to_data + 'batches\\'

ttv_l = ['train_new','test','val']

l_o = 0.1 # percentage/100 of samples to leave out for validation
idx_mf = 2 # most freq. token mapped to 2, 0 reserved for padding, 1 for the out-of-vocab token
oov_idx = 1
min_count = 5
word_vector_dim = 200
batch_size = 128
prepend_title = True

n_jobs = multiprocessing.cpu_count()

# ============================== FUNCTIONS ==============================

# 'atoi' and 'natural_keys' functions taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# function 'my_split' below taken from: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/2135920#2135920
def my_split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def create_assignments(path_to_data,ttv_l,n_jobs):
    '''
    - TODO update description
    - compute the byte offset position of each line, and equally splits them to 'n_jobs' lists
    - parameters:
        - path_to_data: where the files live
        - 'ttv_l': names of the files to process, without extension. E.g., ['train_new','test','val']
                   note that the extension of all files should be '.csv'
        - 'n_jobs': number of cores that will be used later on
    '''
    assignments = dict()
    my_sizes = dict()
    for ttv in ttv_l:
        print('==== going over',ttv, 'set ====')
        my_tells = []
        with open(path_to_data + ttv + '.csv', 'r', encoding='utf8') as infile:
            for line in iter(infile.readline, ''):
                it = infile.tell()
                my_tells.append(it)
        
        my_tells.insert(0,0) # first line starts at byte 0
        my_tells = my_tells[:-1] # last offset position corresponds to empty line!
        print('file features',len(my_tells),'lines')
        my_sizes[ttv] = len(my_tells)
        
        print('creating assignments for', n_jobs, 'cores')
        all_idxs = list(my_split(my_tells,n_jobs))
        
        assert len([elt for sublist in all_idxs for elt in sublist]) == len(my_tells), 'size mismatch!'
    
        # prepend job identifier (integer in [0,n_jobs]) to each sublist
        all_idxs = [[elt_idx] + elt for elt_idx,elt in enumerate(all_idxs)]
        
        assignments[ttv] = all_idxs
        
    return assignments, my_sizes

def tokenize_text(ttv,idxs):
    '''
    - TODO update description
    - this function:
        - splits each line into label, title, and doc
        - tokenizes the doc into sentences and each sentence into words (title is disregarded)
        - creates a word count dict from the training set
        - creates a separate file containing labels (e.g., review scores) (0-based index)
    - parameters:
        - 'ttv': in ['train','test','val']
        - 'idxs': list of byte offset of the lines that should be processed
                  the 1st elt of 'idxs' is the job identifier (for parallel processing)
    '''
    labels = []
    job_nb = idxs[0]
    idxs = idxs[1:]
    tk_ct = dict()
    
    with open(path_to_data + ttv + '.csv', 'r', encoding='utf8') as infile:
        with open(path_to_data + ttv + '_tokenized_'+ str(job_nb) + '.csv', 'a', encoding='utf8') as outfile:
            for idx in idxs:
                infile.seek(idx)
                line = infile.readline()
                line_split = line.split('","')
                label = re.sub('"','',line_split[0])
                if one_based_labels:
                    labels.append(int(label)-1) # keras 'to_categorical' requires 0-based index
                else:
                    labels.append(int(label))
                if prepend_title:
                    txt = line_split[1] + '. ' + line_split[2][:-1]  # prepend title to review
                else:
                    txt = line_split[2][:-1]
                if txt[-1] == '"':
                    txt = txt[:-1]
                txt = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", txt) # turns 'one.two' into 'one . two' taken from: https://stackoverflow.com/questions/27810884/python-how-do-i-separate-punctuation-from-words-by-white-space-leaving-only-on?rq=1
                sents = sent_tokenize(txt)
                txt_to_write = []
                for sent in sents:
                    words = word_tokenize(sent)
                    words = [word.lower() for word in words] # not earlier as case may be necessary for tokenization
                    if ttv == 'train_new': # vocab should be built from training set only!
                        for word in words:
                            tk_ct[word] = tk_ct.get(word, 0) + 1 # increment, creating key if it doesn't already exist
                    txt_to_write.append(' '.join(words)) # separate tokens with white space
                txt_to_write = ' ##S_S## '.join(txt_to_write) + '\n' # separate sentences with '##S_S##'
                outfile.write(txt_to_write)
    
    if ttv == 'train_new':
        with open(path_to_data + 'word_counts_' + str(job_nb) + '.json', 'w', encoding='utf8') as my_file:
            json.dump(tk_ct, my_file, sort_keys=True, indent=4)
    
    with open(path_to_data + ttv + '_labels_'+ str(job_nb) + '.csv', 'a', encoding='utf8') as outfile:
        for label in labels:
            outfile.write(str(label) + '\n')
            
def conv_to_ints(ttv,vocab,idxs):
    '''
    - converts word docs to integer docs
    - parameters:
        - 'ttv': in ['train','test','val']
        - vocab: dict mapping words to integers
        - 'idxs': list of byte offset positions of the lines that should be processed
                  the 1st elt of 'idxs' is the job identifier (for parallel processing)
    '''
    job_nb = idxs[0]
    idxs = idxs[1:]
    
    with open(path_to_data + ttv + '_tokenized.csv', 'r', encoding='utf-8') as infile:
        with open(path_to_data + ttv + '_int_'+ str(job_nb) + '.csv', 'a', encoding='utf8') as outfile:
            for idx in idxs:
                infile.seek(idx)
                line = infile.readline()
                ints = [elt if elt=='##S_S##' else vocab[elt] if elt in vocab else oov_idx for elt in line.split()] 
                to_write = ' '.join([str(elt) for elt in ints]) + '\n'
                outfile.write(to_write)

def get_size(ttv,idxs):
    '''
    - computes the size of each line in the file (# of sents) and prepends it with byte offset position
    - parameters:
        - 'ttv': in ['train','test','val']
        - 'idxs': list of byte offset positions of the lines that should be processed
                  the 1st elt of 'idxs' is the job identifier (for parallel processing)
    '''
    job_nb = idxs[0]
    idxs = idxs[1:]
    
    with open(path_to_data + ttv + '_int.csv', 'r', encoding='utf-8') as infile:
        with open(path_to_data + ttv + '_size_'+ str(job_nb) + '.csv', 'a', encoding='utf8') as outfile:
            for idx in idxs:
                infile.seek(idx)
                line = infile.readline()
                to_write = str(idx) + ' ' + str(len(line.split('##S_S##'))) + '\n'
                outfile.write(to_write)

def write_batch_label(my_assignment,batch_size,my_ttv):
    my_batch_number = my_assignment[0]
    my_assignment = my_assignment[1:]
    
    batch_lines = []
    write_batch = False
    for my_counter, label in enumerate(my_assignment,1):
        batch_lines.append(label)
        
        if my_counter % batch_size == 0:
            write_batch = True
        elif my_counter == len(my_assignment): # the current batch must be the last one, and include the last lines in the file
           write_batch = True
        # for debugging:
        #if write_batch:
        #    print('writing batch number',my_batch_number,len(batch_lines))
        #    my_batch_number += 1
        #    batch_lines = [] 
        #    write_batch = False
          
        if write_batch:
            with open(path_to_batches + my_ttv + '_label_batch_' + str(my_batch_number) + '.txt', 'a', encoding='utf8') as outfile:
                for line in batch_lines:
                    outfile.write(line + '\n')
            my_batch_number += 1
            batch_lines = [] 
            write_batch = False

def write_batch_doc(my_assignment,batch_size,my_ttv):
    my_batch_number = my_assignment[0]
    my_assignment = my_assignment[1:]
    batch_lines = []
    write_batch = False

    with open(path_to_data + my_ttv + '_int.csv', 'r', encoding='utf8') as infile:
        for my_counter,offset in enumerate(my_assignment,1): 
            infile.seek(offset)
            line = infile.readline()
            batch_lines.append(line)
            
            if my_counter % batch_size == 0:
                write_batch = True
            elif my_counter == len(my_assignment): # the current batch must be the last one, and include the last lines in the file
               write_batch = True
            
            if write_batch:
                with open(path_to_batches + my_ttv + '_doc_batch_' + str(my_batch_number) + '.txt', 'a', encoding='utf8') as outfile:
                    for line in batch_lines:
                        outfile.write(line)
                my_batch_number += 1
                batch_lines = [] 
                write_batch = False

# ============================== MAIN ==============================

def main():
    beginning_of_time = time()
    
    print('warning,',n_jobs,'core(s) will be used!')
    
    nltk.download('punkt')
    
    print('creating subfolder to write batch files of size:',batch_size)
    os.mkdir(path_to_batches)
    
    random.seed(7232018)
        
    # save byte offset positions of all lines in 'train.csv'
    my_tells = []
    with open(path_to_data + 'train.csv', 'r', encoding='utf8') as infile:
        for line in iter(infile.readline, ''):
            it = infile.tell()
            my_tells.append(it)
    
    my_tells.insert(0,0) # first line starts at byte 0
    my_tells = my_tells[:-1] # last offset position corresponds to empty line!
    n_train = len(my_tells)
    n_val = round(l_o*n_train)
    
    print('training set features',n_train,'lines')
    print('leaving',n_val,'samples out for validation (',l_o*100,'%)')
    
    val_idxs = list(np.random.choice(np.array(my_tells),size=n_val,replace=False))
    val_idxs.sort() # important that it stays sorted by increasing order!
    
    train_new_idxs = list(set(my_tells).difference(set(val_idxs)))
    train_new_idxs.sort() # important that it stays sorted by increasing order!
    
    assert n_train == len(train_new_idxs) + n_val, 'file size mismatch!'
    
    print('creating new training set as "train_new.csv"')
    
    with open(path_to_data + 'train.csv', 'r', encoding='utf8') as train_old:
        with open(path_to_data + 'train_new.csv', 'a', encoding='utf8') as train_new:
            for it in train_new_idxs:
                train_old.seek(it)
                line = train_old.readline()
                train_new.write(line)
    
    print('creating validation set as "val.csv"')
    
    with open(path_to_data + 'train.csv', 'r', encoding='utf8') as train_old:
        with open(path_to_data + 'val.csv', 'a', encoding='utf8') as val:
            for it in val_idxs:
                train_old.seek(it)
                line = train_old.readline()
                val.write(line)  
    
    assignments, my_sizes = create_assignments(path_to_data,ttv_l,n_jobs)
    print('assignments created')
    
    print('***** doing 1st pass to tokenize and get word counts *****')
    
    for ttv in ttv_l:
        print('==== going over', ttv, 'set ====')
        start = time()      
        all_idxs = assignments[ttv]
        tokenize_text_partial = partial(tokenize_text, ttv)
        print('using', n_jobs, 'cores')
        pool = Pool(processes=n_jobs)
        pool.map(tokenize_text_partial, all_idxs)
        pool.close()
        pool.join()
        
        print(ttv, 'done in', round(time() - start,2))
    
    print('==== combining results ====')
    
    file_list = os.listdir(path_to_data)
    file_list.sort(key=natural_keys)
    
    dict_names = [elt for elt in file_list if re.search('counts_[0-9]+', elt)]
    assert len(dict_names) == len(all_idxs), 'number of dictionaries does not match number of jobs!'  
    
    my_counter_full = Counter()
    for my_file_name in dict_names:
        with open(path_to_data + my_file_name, 'r', encoding='utf8') as my_file:
            tmp = json.load(my_file)
            my_counter_full = my_counter_full + Counter(tmp)
    
    with open(path_to_data + 'word_counts.json', 'w', encoding='utf8') as my_file:
        json.dump(dict(my_counter_full), my_file, sort_keys=True, indent=4)
    
    print('word count dictionaries combined and saved to disk')
    
    tok_names_full = []
    label_names_full = []
    for ttv in ttv_l:
        tok_names = [elt for elt in file_list if re.search(ttv + '_tokenized_[0-9]+', elt)]
        label_names = [elt for elt in file_list if re.search(ttv + '_labels_[0-9]+', elt)]
        assert len(tok_names) == len(all_idxs), 'number of tokenized files does not match number of jobs!'
        assert len(label_names) == len(all_idxs), 'number of label files does not match number of jobs!'
    
        tok_names_full += tok_names
        label_names_full += label_names
        
        with open(path_to_data + ttv + '_tokenized.csv', 'a', encoding='utf8') as outfile:
            for my_file_name in tok_names:
                with open(path_to_data + my_file_name, 'r', encoding='utf8') as infile:
                    for line in infile:
                        outfile.write(line)
        
        with open(path_to_data + ttv + '_labels.csv', 'a', encoding='utf8') as outfile:
            for my_file_name in label_names:
                with open(path_to_data + my_file_name, 'r', encoding='utf8') as infile:
                    for line in infile:
                        outfile.write(line)
    
    print('tokenized and label files combined and saved to disk')      
    
    print('deleting temporary files')
    [os.remove(path_to_data + elt) for elt in dict_names + tok_names_full + label_names_full]
    
    print('tokens appearing less than',min_count,'time(s) will be replaced by special token')
    
    # assign to each word an index based on its frequency in the corpus
    sorted_counts = sorted(my_counter_full.items(), key=operator.itemgetter(1), reverse=True)
    
    word_to_index = dict([(my_t[0],idx) for idx,my_t in enumerate(sorted_counts,idx_mf) if my_t[1]>min_count])
    with open(path_to_data + 'vocab.json', 'w', encoding='utf8') as my_file:
        json.dump(word_to_index, my_file, sort_keys=True, indent=4)
    
    print('vocab created and saved to disk')
    
    print('***** doing 2nd pass to convert textual docs to integer docs *****')
    
    print('updating assignments')
    t_names = [elt + '_tokenized' for elt in ttv_l]
    assignments, my_sizes = create_assignments(path_to_data,t_names,n_jobs)
    
    for ttv_idx,ttv in enumerate(ttv_l):
        print('==== going over', ttv, 'set ====')
        start = time()      
        all_idxs = assignments[t_names[ttv_idx]]
        conv_to_ints_partial = partial(conv_to_ints, ttv, word_to_index)
        print('using', n_jobs, 'cores')
        pool = Pool(processes=n_jobs)
        pool.map(conv_to_ints_partial, all_idxs)
        pool.close()
        pool.join()

        print(ttv, 'done in', round(time() - start,2))
    
    print('==== combining results ====')
    
    file_list = os.listdir(path_to_data)
    file_list.sort(key=natural_keys)
    
    int_names_full = []
    for ttv in ttv_l:
        int_names = [elt for elt in file_list if re.search(ttv + '_int_[0-9]+', elt)]
        assert len(int_names) == len(all_idxs), 'number of integer files does not match number of jobs!'
        int_names_full += int_names
        
        with open(path_to_data + ttv + '_int.csv', 'a', encoding='utf8') as outfile:
            for my_file_name in int_names:
                with open(path_to_data + my_file_name, 'r', encoding='utf8') as infile:
                    for line in infile:
                        outfile.write(line)
    
    print('integer files combined and saved to disk')      
    
    print('deleting temporary files')
    [os.remove(path_to_data + elt) for elt in int_names_full]

    print('***** doing 3rd pass to compute doc size and sort files *****')

    print('updating assignments')
    int_names = [elt + '_int' for elt in ttv_l]
    assignments, my_sizes = create_assignments(path_to_data,int_names,n_jobs)
    
    for ttv_idx,ttv in enumerate(ttv_l):
        print('==== going over', ttv, 'set ====')
        start = time()      
        all_idxs = assignments[int_names[ttv_idx]]
        
        get_size_partial = partial(get_size,ttv)
        print('using', n_jobs, 'cores')
        pool = Pool(processes=n_jobs)
        pool.map(get_size_partial, all_idxs)
        pool.close()
        pool.join()
            
        print(ttv, 'done in', round(time() - start,2))
    
    print('==== combining results ====')
    
    file_list = os.listdir(path_to_data)
    file_list.sort(key=natural_keys)
    
    size_names_full = []
    for ttv in ttv_l:
        size_names = [elt for elt in file_list if re.search(ttv + '_size_[0-9]+', elt)]
        assert len(size_names) == len(all_idxs), 'number of size files does not match number of jobs!'
        size_names_full += size_names
        
        with open(path_to_data + ttv + '_size.csv', 'a', encoding='utf8') as outfile:
            for my_file_name in size_names:
                with open(path_to_data + my_file_name, 'r', encoding='utf8') as infile:
                    for line in infile:
                        outfile.write(line)
    
    print('size files combined and saved to disk')      
    
    print('deleting temporary files')
    [os.remove(path_to_data + elt) for elt in size_names_full]
    
    print('sorting integer and label files by increasing doc size (# of sents) and writing batches')
    
    label_counts = Counter()
    for ttv in ttv_l:
        n_lines = int(my_sizes[ttv + '_int'])
        print('==== going over', ttv, 'set ====')
        # here we exceptionally load entire file to RAM, but each line is very small (just position and size)
        with open(path_to_data + ttv + '_size.csv', 'r', encoding='utf8') as file:
            sizes = file.read().splitlines() # contains byte offset position and size (nb of sents) for each line in the file
    
        size_lists = [[int(eltt) for eltt in elt.split()] for elt in sizes]
        size_lists = [[idx] + elt for idx,elt in enumerate(size_lists)] # prepend line idx (needed to ensure matching with labels)
        size_cats = list(set([elt[2] for elt in size_lists]))
                
        sorted_size_lists = []
        for size_cat in size_cats:
            sub_size_lists = [elt for elt in size_lists if elt[2]==size_cat]
            shuffle(sub_size_lists) # shuffle within each size category
            sorted_size_lists.append(sub_size_lists)
        
        sorted_size_lists = [elt for subl in sorted_size_lists for elt in subl] # flatten
        sorted_line_idxs = [elt[0] for elt in sorted_size_lists] # for labels, later on
        sorted_offsets = [elt[1] for elt in sorted_size_lists]
        
        with open(path_to_data + ttv + '_int_sorted.csv', 'a', encoding='utf8') as outfile:
            with open(path_to_data + ttv + '_int.csv', 'r', encoding='utf8') as infile:
                for offset in sorted_offsets:
                    infile.seek(offset)
                    line = infile.readline()
                    outfile.write(line)
        
        my_assignments = list(my_split(sorted_offsets,n_jobs)) # this does not destroy the ordering
        
        # prepend number of batches that can be built from previous assignments to each assignment so as to preserve batch ordering
        cumul_b = 0
        new_assignments = []
        for ass in my_assignments:
            new_assignments.append([cumul_b] + ass) 
            cumul_b += math.ceil(len(ass)/batch_size) # compute nb of batches that can be obtained from that sublist
        
        write_batch_doc_partial = partial(write_batch_doc,
                                          batch_size=batch_size,
                                          my_ttv=ttv)
        print('using', n_jobs, 'cores')
        start = time()
        pool = Pool(processes=n_jobs)
        pool.map(write_batch_doc_partial, new_assignments)
        pool.close()
        pool.join()

        print('doc batches written in',round(time() - start,2))
        
        # here we again exceptionally load entire file to RAM, but again each line is very small (just label)
        with open(path_to_data + ttv + '_labels.csv', 'r', encoding='utf8') as file:
            labels = file.read().splitlines()
        
        assert len(sizes) == len(labels), 'size and label files have diff. # of rows!'
        assert len(labels) == n_lines, '# of lines and # of labels differ!'
                
        label_counts = label_counts + Counter(labels)
        
        labels_sorted = [labels[idx] for idx in sorted_line_idxs]
        with open(path_to_data + ttv + '_labels_sorted.csv', 'a', encoding='utf8') as file:
            for label in labels_sorted:
                file.write(str(label) + '\n')
        
        # for debugging: labels_sorted = [1,4,2,3,1,2,4,2,3,3,2,2,1,1,4,3,2,1,2,3,4,3]
        # this strategy returns slightly more batches than math.ceil(len(labels_sorted)/n_jobs)
        my_assignments = list(my_split(labels_sorted,n_jobs)) # this does not destroy the ordering
        
        # prepend number of batches that can be built from previous assignments to each assignment so as to preserve batch ordering
        cumul_b = 0
        new_assignments = []
        for ass in my_assignments:
            new_assignments.append([cumul_b] + ass)
            cumul_b += math.ceil(len(ass)/batch_size) # compute nb of batches that can be obtained from that sublist

        write_batch_label_partial = partial(write_batch_label,
                                            batch_size=batch_size,
                                            my_ttv=ttv)
        
        print('using', n_jobs, 'cores')
        start = time()      
        pool = Pool(processes=n_jobs)
        pool.map(write_batch_label_partial, new_assignments)
        pool.close()
        pool.join()
        
        print('label batches written in',round(time() - start,2))
    
    print('label distribution',label_counts)
    print('# of labels=',sum(dict(label_counts).values()))
        
    with open(path_to_data + 'label_distributions.json', 'w', encoding='utf8') as my_file:
        json.dump(label_counts, my_file, sort_keys=True, indent=4)
     
    print('creating data for word2vec from new training set and validation set')
    # one sentence = one line
    
    with open(path_to_data + 'train_new_int_sorted.csv', 'r', encoding='utf8') as infile:
        with open(path_to_data + 'for_w2v.txt', 'a', encoding='utf8') as outfile:
            for line in infile:
                lines = line.split('##S_S##')
                lines = [elt.strip() for elt in lines]
                for l in lines:
                    outfile.write(l + '\n')

    with open(path_to_data + 'val_int_sorted.csv', 'r', encoding='utf8') as infile:
        with open(path_to_data + 'for_w2v.txt', 'a', encoding='utf8') as outfile:
            for line in infile:
                lines = line.split('##S_S##')
                lines = [elt.strip() for elt in lines]
                for l in lines:
                    outfile.write(l + '\n')
    
    print('initializing sentence iterable')
    my_sents = LineSentence(path_to_data + 'for_w2v.txt')
    
    print('learning word vectors')
    assert FAST_VERSION > -1, 'not using cython, word2vec will be too slow!'
    print('using', n_jobs, 'cores')
    
    wv = Word2Vec(sentences=my_sents, 
                  size=word_vector_dim, 
                  min_count=1, # because we have already preprocessed the data
                  workers=n_jobs,
                  iter=10,
                  sg=1,
                  max_vocab_size=2e6) # every 10M words require ~1GB of RAM
    
    wv.save(path_to_data + 'word_vectors.kv') # 'kv' extension stands for 'keyed vectors'
    print('word vectors saved to disk')
    
    # just for sanity check
    with open(path_to_data + 'vocab.json', 'r', encoding='utf8') as my_file:
        vocab = json.load(my_file)
    
    print('length of word2vec vocab (train+val):',len(wv.wv.vocab))
    print('length of vocab (train):',len(vocab))
    assert len(wv.wv.vocab) == len(vocab) + 1, 'word2vec vocab mismatch!' # +1 because vocab doesn't contain the special out-of-vocab token
    
    print('everything done in', round(time() - beginning_of_time,2))    
 
#==============================================================================

if __name__ == "__main__":
    main()
