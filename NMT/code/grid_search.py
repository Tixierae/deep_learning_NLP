import sys
import json

import torch
from torch.utils import data

# = = = = = = = = = = =

path_to_model = './code/'
sys.path.insert(0, path_to_model)
from model import seq2seqModel

path_to_data = './data/'
path_to_save = './models/'

# = = = = = = = = = = =

class Dataset(data.Dataset):
  def __init__(self, pairs):
        self.pairs = pairs

  def __len__(self):
        return len(self.pairs) # total nb of observations

  def __getitem__(self, idx):
        source, target = self.pairs[idx] # one observation
        return torch.LongTensor(source), torch.LongTensor(target)


def load_pairs(train_or_test):
    with open(path_to_data + 'pairs_' + train_or_test + '_ints.txt', 'r', encoding='utf-8') as file:
        pairs_tmp = file.read().splitlines()
    pairs_tmp = [elt.split('\t') for elt in pairs_tmp]
    pairs_tmp = [[[int(eltt) for eltt in elt[0].split()],[int(eltt) for eltt in \
                  elt[1].split()]] for elt in pairs_tmp]
    return pairs_tmp


# = = = = = = = = = = =

pairs_train = load_pairs('train')
pairs_test = load_pairs('test')

with open(path_to_data + 'vocab_source.json','r') as file:
    vocab_source = json.load(file) # word -> index

with open(path_to_data + 'vocab_target.json','r') as file:
    vocab_target = json.load(file) # word -> index

vocab_target_inv = {v:k for k,v in vocab_target.items()} # index -> word

print('data loaded')
    
training_set = Dataset(pairs_train[:5000])
test_set = Dataset(pairs_test[:500])

print('data prepared')

num_layers = 1
bidirectional = True

for att_strategy in ['dot','general','concat']:
    
    hidden_dim_s = 30	
    
    if att_strategy == 'dot':
        hidden_dim_t = 2*hidden_dim_s
    else:
        hidden_dim_t = hidden_dim_s
    
    model = seq2seqModel(vocab_s = vocab_source,
                         source_language = 'english',
                         vocab_t_inv = vocab_target_inv,
                         embedding_dim_s = 40,
                         embedding_dim_t = 40,
                         hidden_dim_s = hidden_dim_s,
                         hidden_dim_t = hidden_dim_t,
                         hidden_dim_att = 20,
                         num_layers = num_layers,
                         bidirectional = bidirectional,
                         att_strategy = att_strategy,
                         padding_token = 0,
                         oov_token = 1,
                         sos_token = 2,
                         eos_token = 3,
                         max_size = 30, # for the decoder, in prediction mode
                         dropout = 0)

    model.fit(training_set, test_set, lr=0.001, batch_size=64, n_epochs=1, patience=5)

    model_name = '_'.join([att_strategy, str(num_layers), str(bidirectional)])
    model.save(path_to_save, model_name)
