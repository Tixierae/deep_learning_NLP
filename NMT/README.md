

# Neural Machine Translation in PyTorch

Here, we provide a compact, fully functional, and well-commented PyTorch implementation of the classical seq2seq model introduced in **Effective Approaches to Attention-based Neural Machine Translation** ([Luong et al. 2015](https://arxiv.org/pdf/1508.04025.pdf)), with support for the three **global attention** mechanisms presented in subsection 3.1 of the paper:


* dot
* general
* concat

<p align="center">
<img src="https://raw.githubusercontent.com/Tixierae/deep_learning_NLP/master/NMT/global_summary_improved.png" alt="" width="550"/>
</p>

Moreover, our implementation provides support for different hyper-parameter combinations:

* stacking vs non-stacking RNN encoder and decoder
* bidirectional vs unidirectional RNN encoder

It also supports model saving (weights + configuration) and re-loading, with ability to resume training from where it was stopped.

We experiment on a small English -> French dataset from http://www.manythings.org/anki/, originally extracted from the [Tatoeba project](https://tatoeba.org/eng/). Our dataset features 136,521 sentence pairs for training and 34,130 pairs for testing. The average size of a source sentence is 7.6 and the average size of
a target sentence is 8.3.

## How to use:

* `pip install -r requirements.txt` 
* `unzip ./data/eng_fr.zip`
* `python preprocessing.py` ()
* `python grid_search.py`


*Note 1:* you might have to install PyTorch manually. If this is the case, install the stable PyTorch version corresponding to CUDA 10 from the [official page](https://pytorch.org/get-started/locally/).

*Note 2:* `preprocessing.py` turns each sentence into a list of integers starting from 4. The integers correspond to indexes in the source and target vocabularies, that are constructed from the training set, and in which the most frequent words have index 4. 0, 1, 2 and 3 are reserved respectively for the padding (`<PAD>`), out-of-vocabulary (`<OOV>`), start-of-sentence (`<SOS>`), and end-of-sentence (`<EOS>`) special tokens.

## TODO

* notebook with results interpretation and visualization
* input-feeding approach
* coverage vector
* beam search decoding
* larger/different dataset

## Thanks :thumbsup:
This implementation was written with valuable code and intellectual help from my labmate [JbRemy](https://github.com/JbRemy)! `#thankyoujb` 

## Cite
If you find some of the code in this repository useful or use it in your own work, please cite
```BibTeX
@article{tixier2018notes,
  title={Notes on Deep Learning for NLP},
  author={Tixier, Antoine J.-P.},
  journal={arXiv preprint arXiv:1808.09772},
  year={2018}
}
```

```
Tixier, A. J. P. (2018). Notes on Deep Learning for NLP. arXiv preprint arXiv:1808.09772.
```
