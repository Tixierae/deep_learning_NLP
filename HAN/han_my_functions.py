import re
import time
import numpy as np
from random import shuffle
from sklearn.metrics import confusion_matrix

from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers import Bidirectional, GRU, CuDNNGRU

def bidir_gru(my_seq,n_units,my_is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    # regardless of whether training is done on GPU, can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if my_is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh', 
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def read_batches(my_batch_names,my_path_to_batches,do_shuffle,do_train,my_max_doc_size_overall,my_max_sent_size_overall,my_n_cats):

    while True:
        
        my_batch_names_docs = [elt for elt in my_batch_names if '_doc_' in elt]
        my_batch_names_labels = [elt for elt in my_batch_names if '_label_' in elt]
        
        # make sure each list is sorted the same way
        my_batch_names_docs.sort(key=natural_keys)
        my_batch_names_labels.sort(key=natural_keys)
        
        assert len(my_batch_names_docs) == len(my_batch_names_labels), 'doc and label # batches differ!'
        
        if do_shuffle:
            my_idxs = list(range(len(my_batch_names_docs)))
            shuffle(my_idxs)
            my_batch_names_docs = [my_batch_names_docs[my_idx] for my_idx in my_idxs]
            my_batch_names_labels = [my_batch_names_labels[my_idx] for my_idx in my_idxs]
        
        for (bnd,bnl) in zip(my_batch_names_docs,my_batch_names_labels):
            
            with open(my_path_to_batches + bnd, 'r', encoding='utf8') as file:
                docs = file.read().splitlines()
            
            with open(my_path_to_batches + bnl, 'r', encoding='utf8') as file:
                label_list = file.read().splitlines()
            
            doc_list = []
            for doc in docs:
                doc = doc.split('##S_S##') # split doc into sentences
                doc = [elt.strip().split() for elt in doc] # split sentences into words
                doc_list.append(doc)
            
            max_doc_size = min(max([len(d) for d in doc_list]),my_max_doc_size_overall)
            max_sent_size = min(max([len(s) for d in doc_list for s in d]),my_max_sent_size_overall)
            
            # padding (doc level) - add or remove sentences
            doc_list = [d+[[0]*max_sent_size]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in doc_list] 
            # padding (sent level) - add or remove words
            doc_list = [[s+[0]*(max_sent_size-len(s)) if len(s)<max_sent_size else s[:max_sent_size] for s in d] for d in doc_list] 
            doc_array = np.array(doc_list,dtype='int64')
            label_array = np.array(label_list,dtype='int64')
            label_array = to_categorical(label_array,num_classes=my_n_cats)
            
            if do_train:
                yield(doc_array,label_array)
            else:
                yield(doc_array)

class PerClassAccHistory(Callback):
    '''
    a note about the confusion matrix:
    Cij = nb of obs known to be in group i and predicted to be in group j. So:
    - the nb of right predictions is given by the diagonal
    - the total nb of observations for a group is given by summing the corresponding row
    - the total nb of predictions for a group is given by summing the corresponding col
    accuracy is (nb of correct preds)/(total nb of preds)
    # https://developers.google.com/machine-learning/crash-course/classification/accuracy
    '''
    def __init__(self, my_n_cats, my_rd, my_n_steps):
        self.my_n_cats = my_n_cats
        self.my_rd = my_rd
        self.my_n_steps = my_n_steps

    def on_train_begin(self, logs={}):
        self.per_class_accuracy = []
     
    def on_epoch_end(self, epoch, logs={}):
        cmat = np.zeros(shape=(self.my_n_cats,self.my_n_cats))
        for repeat in range(self.my_n_steps):
            docs, labels = self.my_rd.__next__()
            preds_floats = self.model.predict(docs)
            y_pred = np.argmax(np.array(preds_floats),axis=1)
            y_true = np.argmax(labels,axis=1)
            cmat = np.add(cmat,confusion_matrix(y_true, y_pred))
            if repeat % round(self.my_n_steps/5) == 0:
                print(repeat)
        accs = list(np.round(1e2*cmat.diagonal()/cmat.sum(axis=0),2))
        self.per_class_accuracy.append(accs)

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()
    
    def on_epoch_end(self, batch, logs={}):
        self.times.append(round(time() - self.epoch_time_start,2))

class LossHistory(Callback):
    '''
    records the average loss on the full *training* set so far
    the loss returned by logs is just that of the current batch!
    '''
    def on_train_begin(self, logs={}):
        self.losses = []
        self.loss_avg = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(round(float(logs.get('loss')),6))
        self.loss_avg.append(round(np.mean(self.losses,dtype=np.float64),6))
    
    def on_epoch_end(self, batch, logs={}):
        self.losses = []
    
class AccHistory(Callback):
    '''
    records the average accuracy on the full *training* set so far
    the accuracy returned by logs is just that of the current batch!
    '''
    def on_train_begin(self, logs={}):
        self.accs = []
        self.acc_avg = []

    def on_batch_end(self, batch, logs={}):
        self.accs.append(round(1e2*float(logs.get('acc')),4))
        self.acc_avg.append(round(np.mean(self.accs,dtype=np.float64),4))
    
    def on_epoch_end(self, batch, logs={}):
        self.accs = []

class LRHistory(Callback):
    ''' records the current learning rate'''
    def on_train_begin(self, logs={}):
       self.lrs = []
    
    def on_batch_end(self, batch, logs={}):
        my_lr = K.eval(self.model.optimizer.lr)
        self.lrs.append(my_lr)

class MTHistory(Callback):
    ''' records the current momentum'''
    def on_train_begin(self, logs={}):
       self.mts = []
    
    def on_batch_end(self, batch, logs={}):
        my_mt = K.eval(self.model.optimizer.momentum)
        self.mts.append(my_mt)