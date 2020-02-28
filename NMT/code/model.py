import json
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import corpus_bleu
'''
hyp = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
ref = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
list_of_references = [[ref]]
hypotheses = [hyp]
print(corpus_bleu(list_of_references, hypotheses))
'''

test_sents = ['I am a student.',
              'I have a red car.',  # adj-verb inversion
              'Kids love playing video games.',
              'This river is full of fish.', # plein vs pleine (accord)
              'The fridge is full of food.',
              'The cat fell asleep on the mat.',
              'my brother likes pizza.', # pizza is translated to 'la pizza'
              'I did not mean to hurt you', # translation of mean in context
              'She is so mean',
              'Help me pick out a tie to go with this suit!', # more involved sentences
              "I can't help but smoking weed",
              'The kids were playing hide and seek',
              'The cat fell asleep in front of the fireplace']

from tqdm import tqdm

from nltk import word_tokenize

class Encoder(nn.Module):
    '''
    to be passed the entire source sequence at once
    we use padding_idx in nn.Embedding so that the padding vector does not receive gradient and stays always at zero
    https://pytorch.org/docs/stable/nn.html#{embedding,dropout,gru}
    '''
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, padding_idx, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            hidden_dim = int(hidden_dim/2)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, input):
        word_vectors = self.embedding(input)
        word_vectors = self.dropout(word_vectors)
        hs,_ = self.rnn(word_vectors) # (seq,batch,embedding_dim). We are only interested in the first element of the output (annotations of the top layer of the stacking RNN)
        
        return hs


class seq2seqAtt(nn.Module):
    '''
    returns context vector given current target annotation (decoder) and all source annotations (encoder)
    strategy can be 'dot', 'general', or 'concat', as in subsection 3.1 of Luong et al. 2015
    (see: https://arxiv.org/pdf/1508.04025.pdf)
    '''
    
    def __init__(self, hidden_dim, hidden_dim_s, hidden_dim_t, strategy):
        super(seq2seqAtt, self).__init__()
        self.strategy = strategy
        assert strategy in ['dot','general','concat'], "'strategy' must be in ['dot','general','concat']"
        
        if strategy == 'dot':
            assert hidden_dim_s == hidden_dim_t, "with 'dot' strategy, source and target hidden dims must be equal!"
            
        elif strategy == 'general':
            self.ff_general = nn.Linear(hidden_dim_t, hidden_dim_s)
            
        elif strategy == 'concat':
            self.ff_concat = nn.Linear(hidden_dim_s+hidden_dim_t, hidden_dim)
            self.ff_score = nn.Linear(hidden_dim, 1, bias=False) # dot product with trainable vector
        
    def forward(self, target_h, source_hs):
        
        if self.strategy in ['dot','general']:
            source_hs = source_hs.permute(1,0,2) # (seq,batch,hidden_dim_s) -> (batch,seq,hidden_dim_s)
        
        if self.strategy == 'dot':
            # with this strategy, no trainable parameters are involved
            # here, feat = hidden_dim_t = hidden_dim_s
            target_h = target_h.permute(1,2,0) # (1,batch,feat) -> (batch,feat,1)
            dot_product = torch.matmul(source_hs, target_h) # (batch,seq,feat) * (batch,feat,1) -> (batch,seq,1)
            scores = dot_product.permute(1,0,2) # -> (seq,batch,1)
            
        elif self.strategy == 'general':
            target_h = target_h.permute(1,0,2) # (1,batch,hidden_dim_t) -> (batch,1,hidden_dim_t)
            output = self.ff_general(target_h) #  -> (batch,1,hidden_dim_s)
            output = output.permute(0,2,1) # -> (batch,hidden_dim_s,1)
            dot_product = torch.matmul(source_hs, output) # (batch,seq,hidden_dim_s) * (batch,hidden_dim_s,1) -> (batch,seq,1)
            scores = dot_product.permute(1,0,2) # -> (seq,batch,1)
            
        elif self.strategy == 'concat':
            target_h_rep = target_h.repeat(source_hs.size(0),1,1) # (1,batch,hidden_dim_s) -> (seq,batch,hidden_dim_s)
            concat_output = self.ff_concat(torch.cat((target_h_rep,source_hs),-1)) # (seq,batch,hidden_dim_s+hidden_dim_t) -> (seq,batch,hidden_dim)
            scores = self.ff_score(torch.tanh(concat_output)) # -> (seq,batch,1)
            source_hs = source_hs.permute(1,0,2)  # (seq,batch,hidden_dim_s) -> (batch,seq,hidden_dim_s)
                
        scores = scores.squeeze(dim=2) # (seq,batch,1) -> (seq,batch). We specify a dimension, because we don't want to squeeze the batch dim in case batch size is equal to 1
        norm_scores = torch.softmax(scores,0) # sequence-wise normalization
        source_hs_p = source_hs.permute((2,1,0)) # (batch,seq,hidden_dim_s) -> (hidden_dim_s,seq,batch)
        weighted_source_hs = (norm_scores * source_hs_p) # (seq,batch) * (hidden_dim_s,seq,batch) -> (hidden_dim_s,seq,batch) (we use broadcasting here - the * operator checks from right to left that the dimensions match)
        ct = torch.sum(weighted_source_hs.permute((1,2,0)),0,keepdim=True) # (hidden_dim_s,seq,batch) -> (seq,batch,hidden_dim_s) -> (1,batch,hidden_dim_s); we need keepdim as sum squeezes by default 
        
        return ct


class Decoder(nn.Module):
    '''to be used one timestep at a time
       https://pytorch.org/docs/stable/nn.html#gru'''
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim_t, hidden_dim_s, num_layers, padding_idx, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.dropout_1 = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim, hidden_dim_t, num_layers, dropout=dropout)
        self.ff_concat = nn.Linear(hidden_dim_s + hidden_dim_t, hidden_dim_t)
        self.dropout_2 = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_dim_t, vocab_size)
    
    def forward(self, input, source_context, h):
        word_vector = self.embedding(input) # (1,batch) -> (1,batch,embedding_dim)
        word_vector = self.dropout_1(word_vector)
        h_top, h_all = self.rnn(word_vector, h) # output is (1,batch,hidden_dim_t), (num_layers,batch,hidden_dim_t); ([0]: top stacking RNN layer, [1]: all stacking RNN layers)
        h_tilde = torch.tanh(self.ff_concat(torch.cat((source_context, h_top), -1))) # (1,batch,hidden_dim_s+hidden_dim_t) -> (1,batch,hidden_dim_t). This corresponds to Eq. 5 in Luong et al. 2015
        h_tilde = self.dropout_2(h_tilde)
        prediction = self.final(h_tilde) # (1,batch,feat) -> (1,batch,vocab) note that the prediction is not normalized at this time (it is just a vector of logits)
        
        return prediction, h_all


class seq2seqModel(nn.Module):
    '''full seq2seq model a la Luong et al. 2015'''
    
    ARGS = ['vocab_s', 'source_language', 'vocab_t_inv', 'embedding_dim_s', 'embedding_dim_t',
            'hidden_dim_s', 'hidden_dim_t', 'hidden_dim_att', 'num_layers', 'bidirectional',
            'att_strategy', 'padding_token', 'oov_token', 'sos_token', 'eos_token', 
            'max_size','dropout']
    
    def __init__(self, vocab_s, source_language, vocab_t_inv, embedding_dim_s, embedding_dim_t, hidden_dim_s, hidden_dim_t, hidden_dim_att, num_layers, bidirectional, att_strategy, padding_token, oov_token, sos_token, eos_token, max_size, dropout):
        super(seq2seqModel, self).__init__()
        self.vocab_s = vocab_s
        self.source_language = source_language
        self.vocab_t_inv = vocab_t_inv
        self.embedding_dim_s = embedding_dim_s # if bidirectional, this will be the number of features AFTER concatenating the forward and backward RNNs!
        self.embedding_dim_t = embedding_dim_t
        self.hidden_dim_s = hidden_dim_s
        self.hidden_dim_t = hidden_dim_t
        self.hidden_dim_att = hidden_dim_att
        self.num_layers = num_layers
        self.bidirectional = bidirectional # applies to the encoder only
        self.att_strategy = att_strategy # should be in ['none','dot','general','concat']
        self.padding_token = padding_token
        self.oov_token = oov_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_size = max_size # only used when decoding
        self.dropout = dropout
        self.logs = dict()
        
        self.max_source_idx = max(list(vocab_s.values()))
        print('max source index',self.max_source_idx)
        print('source vocab size',len(vocab_s))
        
        self.max_target_idx = max([int(elt) for elt in list(vocab_t_inv.keys())])
        print('max target index',self.max_target_idx)
        print('target vocab size',len(vocab_t_inv))
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = Encoder(self.max_source_idx+1, self.embedding_dim_s, self.hidden_dim_s, self.num_layers,self.bidirectional, self.padding_token, self.dropout).to(self.device)
        
        self.decoder = Decoder(self.max_target_idx+1, self.embedding_dim_t, self.hidden_dim_t, self.hidden_dim_s, self.num_layers, self.padding_token, self.dropout).to(self.device)
        
        if not self.att_strategy == 'none':
            self.att_mech = seq2seqAtt(self.hidden_dim_att, self.hidden_dim_s, self.hidden_dim_t, self.att_strategy).to(self.device)
    
    def my_pad(self, my_list):
        '''my_list looks like: [(seq_s_1,seq_t_1),...,(seq_s_n,seq_t_n)], where n is batch size.
        Each tuple is a pair of source and target sequences
        the <eos> token is appended to each sequence before padding
        https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence'''
        
        batch_source = pad_sequence([torch.cat((elt[0], torch.LongTensor([self.eos_token]))) for elt in my_list], batch_first=True, padding_value=self.padding_token)
        batch_target = pad_sequence([torch.cat((elt[1], torch.LongTensor([self.eos_token]))) for elt in my_list], batch_first=True, padding_value=self.padding_token)
        
        batch_source = batch_source.transpose(1,0).to(self.device) # (batch,seq) -> (seq,batch) (for RNN)
        batch_target = batch_target.transpose(1,0).to(self.device)
        
        return batch_source, batch_target
        
    
    def forward(self, input, max_size, is_prod):
        
        if is_prod: 
            input = input.unsqueeze(1) # (seq) -> (seq,1) 1D input. In production/API mode, we receive just one sentence as input
        
        current_batch_size = input.size(1)
    
        source_hs = self.encoder(input) # (seq,batch) or (seq,1) -> (seq,batch,hidden_dim_s) or (seq,1,hidden_dim_s)
        
        # = = = decoder part (one timestep at a time)  = = =
        
        # = init =
        target_h_top = torch.zeros(size=(1, current_batch_size, self.hidden_dim_t)).to(self.device) # (1,batch,hidden_dim_t)
        target_h_all = torch.zeros(size=(self.num_layers, current_batch_size, self.hidden_dim_t)).to(self.device) # (num_layers,batch,hidden_dim_t)
        target_input = torch.LongTensor([self.sos_token]).repeat(current_batch_size).unsqueeze(0).to(self.device) # (1,batch)
        # = / init =
        
        pos = 0 # counter for the nb of decoding steps that have been performed
        eos_counter = 0
        logits = []
        
        while True:
            
            if not self.att_strategy == 'none':
                source_context = self.att_mech(target_h_top, source_hs) # (1,batch,hidden_dim_s)
            else:
                source_context = source_hs[-1,:,:].unsqueeze(0) # (1,batch,hidden_dim_s) last hidden state of encoder
            
            prediction, target_h_all = self.decoder(target_input, source_context, target_h_all) # (1,batch,vocab), (num_layers,batch,hidden_dim_t)
            
            logits.append(prediction) # we keep the logits to compute the loss later on
            
            target_input = prediction.argmax(-1)
            
            eos_counter += torch.sum(target_input==self.eos_token).item()
            
            pos += 1
            if pos>=max_size or (eos_counter == current_batch_size and is_prod):
                break
        
        to_return = torch.cat(logits,0) # list of tensors of shape (batch,vocab) -> (seq,batch,vocab)
        
        if is_prod:
            to_return = to_return.squeeze(dim=1) # (seq,vocab)
        
        return to_return
    
    
    def fit(self, trainingDataset, testDataset, lr, batch_size, n_epochs, patience):
        
        parameters = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = optim.Adam(parameters, lr=lr)
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_token) # the softmax is inside the loss!
        
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        # we pass a collate function to perform padding on the fly, within each batch
        # this is better than truncation/padding at the dataset level
        # TODO bucketing
        train_loader = data.DataLoader(trainingDataset, batch_size=batch_size, 
                                       shuffle=True, collate_fn=self.my_pad) # returns (batch,seq)
        
        test_loader = data.DataLoader(testDataset, batch_size=512, collate_fn=self.my_pad)
        
        tdqm_dict_keys = ['loss', 'test loss', 'test BLEU']
        tdqm_dict = dict(zip(tdqm_dict_keys,[0.0]*len(tdqm_dict_keys)))
        
        patience_counter = 1
        patience_loss = 99999
        it_times = []
        
        for epoch in range(n_epochs):
            
            logs_epoch = {'loss':[],'test_loss':[],'test_BLEU':[]}
            
            with tqdm(total=len(train_loader), unit_scale=True, postfix=tdqm_dict, desc="Epoch : %i/%i" % (epoch, n_epochs-1), ncols=100) as pbar:
                
                for loader_idx, loader in enumerate([train_loader, test_loader]):
                    
                    total_loss = 0
                    total_bleu = 0
                    
                    if loader_idx == 0:
                        self.train()
                    else:
                        self.eval() # deactivate dropout etc.
                    
                    for i, (batch_source, batch_target) in enumerate(loader):
                        
                        start_time = time()
                        
                        is_prod = len(batch_source.shape)==1 # if False, 2D input (seq,batch)
                        
                        if is_prod:
                            max_size = self.max_size
                            self.eval()
                        else:
                            max_size = batch_target.size(0) # in train mode, no need to continue generating after the length of the longest ground truth sequence has been exceeded
                        
                        unnormalized_logits = self.forward(batch_source, max_size, is_prod) # (seq,batch,vocab)
                        
                        # = = loss = = 
                        batch_loss = criterion(unnormalized_logits.flatten(end_dim=1), batch_target.flatten())
                        # with flatten, (see https://pytorch.org/docs/stable/nn.html#flatten), we turn unnormalized_logits into (seq*batch,vocab). We compare it with (seq*batch,1)
                        total_loss += batch_loss.item()
                        updated_loss = total_loss/(i+1)
                        tdqm_dict[tdqm_dict_keys[loader_idx]] = updated_loss
                        
                        if loader_idx == 1:
                            logs_epoch['test_loss'].append(round(updated_loss,4))
                        else:
                            # save every 100 batches
                            if i % int(len(trainingDataset)/100) == 0:
                                logs_epoch['loss'].append(round(updated_loss,4))
                        
                        # = = BLEU = =
                        if loader_idx == 1:
                            refs = self.for_bleu(batch_target, True)
                            hyps = self.for_bleu(unnormalized_logits, False)
                            batch_bleu = corpus_bleu(refs, hyps)
                            total_bleu += batch_bleu
                            updated_bleu = total_bleu/(i+1)
                            tdqm_dict['test BLEU'] = updated_bleu
                            logs_epoch['test_BLEU'].append(round(100*updated_bleu,2))
                        
                        pbar.set_postfix(tdqm_dict)
                        
                        if loader_idx == 0:
                            optimizer.zero_grad() # flush gradient attributes
                            batch_loss.backward() # compute gradients
                            optimizer.step() # update parameters of the model
                            pbar.update(1)
                        
                        it_times.append(round(time() - start_time,2))
                        
            self.logs['epoch_' + str(epoch+1)] = logs_epoch
            
            if total_loss > patience_loss:
                patience_counter += 1
            else:
                patience_loss = total_loss
                patience_counter = 1 # reset
            
            if patience_counter>patience:
                break
        
        self.test_toy(test_sents) 
        
        self.logs['avg_time_it'] = round(np.mean(it_times),4)
        self.logs['n_its'] = len(it_times)
       
        
    def sourceNl_to_ints(self, source_nl):
        '''converts natural language source sentence to source integers'''
        source_nl_clean = source_nl.lower().replace("'",' ').replace('-',' ')
        source_nl_clean_tok = word_tokenize(source_nl_clean, self.source_language)
        source_ints = [int(self.vocab_s[elt]) if elt in self.vocab_s else \
                       self.oov_token for elt in source_nl_clean_tok]
        
        source_ints = torch.LongTensor(source_ints).to(self.device)
        return source_ints 
    
    def targetInts_to_nl(self, target_ints):
        '''converts integer target sentence to target natural language'''
        return ['<PAD>' if elt==self.padding_token else '<OOV>' if elt==self.oov_token \
                else '<EOS>' if elt==self.eos_token else '<SOS>' if elt==self.sos_token\
                else self.vocab_t_inv[elt] for elt in target_ints]
    
    def predict(self, source_nl):
        source_ints = self.sourceNl_to_ints(source_nl)
        logits = self.forward(source_ints, self.max_size, True) # (seq) -> (<=max_size,vocab)
        target_ints = logits.argmax(-1).squeeze() # (<=max_size,1) -> (<=max_size)
        target_ints_list = target_ints.tolist()
        if not isinstance(target_ints_list, list):
            target_ints_list = [target_ints_list]
        target_nl = self.targetInts_to_nl(target_ints_list)
        return ' '.join(target_nl)
    
    def test_toy(self, source_sents):
        for elt in source_sents:
            print('= = = = = \n','%s -> %s' % (elt, self.predict(elt)))
    
    def for_bleu(self, logits_or_ints, is_ref):
        if not is_ref:
            # here, logits_or_ints contains the logits
            targets_ints = logits_or_ints.argmax(-1).permute(1,0) # (seq,batch,vocab) -> (seq,batch) -> (batch,seq)
        else:
            # here, we directly have the indexes (integers), as (seq,batch). We just turn it into (batch,seq)
            targets_ints = logits_or_ints.permute(1,0)
        
        sents = [self.targetInts_to_nl(elt.tolist()) for elt in targets_ints] # (batch,seq)
        # remove all words after the first occurrence of '<EOS>'
        sents = [elt[:elt.index('<EOS>')+1] if '<EOS>' in elt else elt for elt in sents]
        if is_ref:
            sents = [[elt] for elt in sents] # BLEU expects references to be a list of lists of lists
        return sents

    def save(self, path_to_save, model_name):
        attrs = {attr:getattr(self,attr) for attr in self.ARGS}
        attrs['state_dict'] = self.state_dict()
        torch.save(attrs, path_to_save + model_name + '.pt')
        
        args_save = [elt for elt in self.ARGS if 'vocab' not in elt]
        params = {attr:getattr(self,attr) for attr in args_save}
        
        with open(path_to_save + 'params_' + model_name + '.json', 'w') as file:
            json.dump(params, file, sort_keys=True, indent=4)
        
        with open(path_to_save + 'logs_' + model_name + '.json', 'w') as file:
            json.dump(self.logs, file, sort_keys=True, indent=4)
    
    @classmethod # a class method does not see the inside of the class (a static method does not take self as first argument)
    def load(cls,path_to_file):
        attrs = torch.load(path_to_file, map_location=lambda storage, loc: storage) # allows loading on CPU a model trained on GPU, see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/6
        state_dict = attrs.pop('state_dict')
        new = cls(**attrs) # * list and ** names (dict) see args and kwargs
        new.load_state_dict(state_dict)
        return new
